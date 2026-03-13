"""
m17_model.py
────────────
MNIST ONNX 모델의 로드 · 전처리 · 추론을 캡슐화한 클래스 모듈입니다.

Usage
-----
    from m17_model import MnistModel

    model = MnistModel(model_path)          # 인스턴스 생성 (모델 로드)
    pred, probs, arr28 = model.doPredict(pil_image)  # 추론
"""

import os
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageFilter
from scipy import ndimage


class MnistModel:
    """MNIST ONNX 모델을 래핑하는 추론 클래스.

    Parameters
    ----------
    model_path : str
        ONNX 모델 파일 경로 (.onnx)

    Raises
    ------
    FileNotFoundError
        지정한 경로에 모델 파일이 없을 경우 발생합니다.
    """

    def __init__(self, model_path: str) -> None:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        self.__Session = ort.InferenceSession(model_path)
        input_meta = self.__Session.get_inputs()[0]
        self.__InputName = input_meta.name
        self.__InputType = input_meta.type
        self.__InputShape = input_meta.shape
        self.__OutputName = self._selectOutputName()
        self.__DebugInfo = {}

    # ── 내부 전처리 ──────────────────────────────────────────────────────────

    @staticmethod
    def _preprocess(arr: np.ndarray) -> np.ndarray:
        """MNIST 학습 시 사용한 전처리를 재현합니다.

        1. 바운딩 박스 크롭 (빈 여백 제거)
        2. 비율 유지하며 20×20 에 맞게 리사이즈
        3. 4px 패딩 → 28×28 캔버스 중앙 배치
        4. 무게중심(center of mass) 기반 중앙 정렬

        Parameters
        ----------
        arr : np.ndarray  shape=(H, W), dtype=float32, range=[0, 255]
            검정 배경(0) + 흰 글씨(밝음) 형태의 그레이스케일 배열

        Returns
        -------
        np.ndarray  shape=(28, 28), dtype=float32
        """
        rows = np.any(arr > 10, axis=1)
        cols = np.any(arr > 10, axis=0)
        if not rows.any():
            return np.zeros((28, 28), dtype=np.float32)

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        cropped = arr[rmin: rmax + 1, cmin: cmax + 1]

        # 20×20 내부에 비율 유지 리사이즈
        h, w = cropped.shape
        scale = 20.0 / max(h, w)
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        pil_crop = Image.fromarray(cropped.astype(np.uint8)).resize(
            (new_w, new_h), Image.LANCZOS
        )
        resized = np.array(pil_crop, dtype=np.float32)

        # 28×28 캔버스 중앙 배치
        canvas = np.zeros((28, 28), dtype=np.float32)
        row_off = (28 - new_h) // 2
        col_off = (28 - new_w) // 2
        canvas[row_off: row_off + new_h, col_off: col_off + new_w] = resized

        # 무게중심 기반 중앙 정렬
        cy, cx = ndimage.center_of_mass(canvas)
        shift_y = int(round(14 - cy))
        shift_x = int(round(14 - cx))
        canvas = ndimage.shift(canvas, (shift_y, shift_x), mode="constant", cval=0)

        return canvas

    @staticmethod
    def _preprocessDirect(arr: np.ndarray) -> np.ndarray:
        """모델 README에 맞춰 전체 이미지를 바로 28x28로 축소합니다."""
        pil_img = Image.fromarray(arr.astype(np.uint8)).resize((28, 28), Image.LANCZOS)
        return np.array(pil_img, dtype=np.float32)

    @staticmethod
    def _preprocessBinary(arr: np.ndarray) -> np.ndarray:
        """흰 글씨를 더 또렷하게 하기 위해 이진화 후 중심 정렬 전처리를 적용합니다."""
        binary = np.where(arr > 32, 255.0, 0.0).astype(np.float32)
        return MnistModel._preprocess(binary)

    def _selectOutputName(self) -> str:
        """분류 점수(10 클래스)로 보이는 출력 노드를 우선 선택합니다."""
        outputs = self.__Session.get_outputs()
        for out in outputs:
            shape = out.shape
            if isinstance(shape, list) and shape:
                if shape[-1] == 10 or shape == [1, 10]:
                    return out.name
        return outputs[0].name

    def _toModelInput(self, arr: np.ndarray, arr_norm: np.ndarray) -> np.ndarray:
        """모델 입력 dtype/shape에 맞춰 텐서를 생성합니다."""
        if "uint8" in self.__InputType:
            base = arr.astype(np.uint8)
        elif "int8" in self.__InputType:
            base = np.clip(arr - 128.0, -128, 127).astype(np.int8)
        else:
            base = arr_norm.astype(np.float32)

        rank = len(self.__InputShape)
        if rank == 4:
            # NCHW: [N, 1, 28, 28] 또는 NHWC: [N, 28, 28, 1]
            if self.__InputShape[1] == 1:
                return base.reshape(1, 1, 28, 28)
            return base.reshape(1, 28, 28, 1)
        if rank == 3:
            # [N, 28, 28]
            return base.reshape(1, 28, 28)
        if rank == 2:
            # [N, 784]
            return base.reshape(1, -1)
        return base.reshape(1, 1, 28, 28)

    def _reshapeByInputShape(self, base: np.ndarray) -> np.ndarray:
        """입력 rank/채널 축에 맞춰 텐서를 재구성합니다."""
        rank = len(self.__InputShape)
        if rank == 4:
            if self.__InputShape[1] == 1:
                return base.reshape(1, 1, 28, 28)
            return base.reshape(1, 28, 28, 1)
        if rank == 3:
            return base.reshape(1, 28, 28)
        if rank == 2:
            return base.reshape(1, -1)
        return base.reshape(1, 1, 28, 28)

    @staticmethod
    def _toProbabilities(scores: np.ndarray) -> np.ndarray:
        """출력이 이미 확률이면 그대로 사용하고, 아니면 소프트맥스를 적용합니다."""
        v = np.asarray(scores, dtype=np.float32).reshape(-1)
        if v.size == 10 and np.all(v >= 0.0) and np.isclose(v.sum(), 1.0, atol=1e-3):
            return v
        e = np.exp(v - np.max(v))
        return e / np.sum(e)

    @staticmethod
    def _getScoreMargin(raw_scores: np.ndarray) -> float:
        """raw score 기준 top1-top2 margin을 반환합니다."""
        values = np.sort(np.asarray(raw_scores, dtype=np.float32).reshape(-1))
        if values.size < 2:
            return 0.0
        return float(values[-1] - values[-2])

    def _doInferDetailed(self, model_input: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """단일 입력 텐서로 추론하고 raw 출력과 확률 벡터를 반환합니다."""
        raw_output = self.__Session.run(
            [self.__OutputName], {self.__InputName: model_input}
        )[0]
        probs = self._toProbabilities(raw_output)
        return np.asarray(raw_output, dtype=np.float32).reshape(-1), probs

    def _doInferBestEffort(self, arr: np.ndarray, arr_norm: np.ndarray) -> tuple[np.ndarray, dict]:
        """입력 스케일이 불명확한 모델을 위해 후보 스케일을 비교하여 최적 확률을 선택합니다."""
        # 양자화 입력은 dtype이 명확하므로 기존 경로를 사용
        if "uint8" in self.__InputType or "int8" in self.__InputType:
            raw_scores, probs = self._doInferDetailed(self._toModelInput(arr, arr_norm))
            margin = self._getScoreMargin(raw_scores)
            return probs, {
                "selected_scale": "quantized-native",
                "raw_scores": raw_scores.tolist(),
                "confidence": float(np.max(probs)),
                "margin": margin,
                "scale_candidates": [
                    {
                        "name": "quantized-native",
                        "confidence": float(np.max(probs)),
                        "margin": margin,
                        "prediction": int(np.argmax(probs)),
                        "raw_scores": raw_scores.tolist(),
                    }
                ],
            }

        # float 계열은 0~1, 0~255 모두 시도해 더 확신도 높은 결과를 선택
        arr_std = ((arr / 255.0) - 0.1307) / 0.3081
        candidate_specs = [
            ("normalized-0-1", self._reshapeByInputShape(arr_norm.astype(np.float32))),
            ("raw-0-255", self._reshapeByInputShape(arr.astype(np.float32))),
            ("mnist-standardized", self._reshapeByInputShape(arr_std.astype(np.float32))),
        ]
        candidate_results = []
        for scale_name, model_input in candidate_specs:
            raw_scores, probs = self._doInferDetailed(model_input)
            margin = self._getScoreMargin(raw_scores)
            candidate_results.append(
                {
                    "name": scale_name,
                    "probs": probs,
                    "raw_scores": raw_scores,
                    "confidence": float(np.max(probs)),
                    "margin": margin,
                    "prediction": int(np.argmax(probs)),
                }
            )

        best_idx = int(np.argmax([item["margin"] for item in candidate_results]))
        best = candidate_results[best_idx]
        return best["probs"], {
            "selected_scale": best["name"],
            "raw_scores": best["raw_scores"].tolist(),
            "confidence": best["confidence"],
            "margin": best["margin"],
            "scale_candidates": [
                {
                    "name": item["name"],
                    "confidence": item["confidence"],
                    "margin": item["margin"],
                    "prediction": item["prediction"],
                    "raw_scores": item["raw_scores"].tolist(),
                }
                for item in candidate_results
            ],
        }

    def _doPredictFromArray(
        self, arr: np.ndarray, preprocess_name: str
    ) -> tuple[int, np.ndarray, np.ndarray, dict]:
        """전처리된 28x28 배열 하나에 대해 추론을 수행합니다."""
        arr_norm = arr / 255.0
        probs, scale_debug = self._doInferBestEffort(arr, arr_norm)
        debug_info = {
            "preprocess_name": preprocess_name,
            "preprocess_min": float(np.min(arr)),
            "preprocess_max": float(np.max(arr)),
            "preprocess_mean": float(np.mean(arr)),
            "selected_scale": scale_debug["selected_scale"],
            "raw_scores": scale_debug["raw_scores"],
            "confidence": scale_debug["confidence"],
            "margin": scale_debug["margin"],
            "scale_candidates": scale_debug["scale_candidates"],
        }
        return int(np.argmax(probs)), probs, arr_norm, debug_info

    # ── 공개 메서드 ──────────────────────────────────────────────────────────

    def doPredict(self, pil_image: Image.Image) -> tuple[int, np.ndarray, np.ndarray]:
        """PIL 이미지를 받아 MNIST 예측 결과를 반환합니다.

        Parameters
        ----------
        pil_image : PIL.Image.Image
            입력 이미지 (크기·모드 무관; 내부적으로 28×28 그레이스케일로 변환)

        Returns
        -------
        pred_class : int
            예측된 숫자 클래스 (0 ~ 9)
        probs : np.ndarray  shape=(10,)
            각 클래스에 대한 소프트맥스 확률
        arr28 : np.ndarray  shape=(28, 28), dtype=float32, range=[0, 1]
            실제 모델에 입력된 전처리 이미지 (시각화용)
        """
        # 그레이스케일 변환 + 가우시안 블러 (잡음 제거)
        img = pil_image.convert("L").filter(ImageFilter.GaussianBlur(radius=1))
        arr = np.array(img, dtype=np.float32)

        # 캔버스 배경이 흰색(255)이면 반전 (MNIST: 검정 배경 + 흰 글씨)
        if arr.mean() > 127:
            arr = 255.0 - arr

        candidates = [
            ("centered", self._preprocess(arr)),
            ("direct-resize", self._preprocessDirect(arr)),
            ("binary-centered", self._preprocessBinary(arr)),
        ]

        results = [
            self._doPredictFromArray(candidate, preprocess_name)
            for preprocess_name, candidate in candidates
        ]
        best_idx = int(np.argmax([result[3]["margin"] for result in results]))
        best_result = results[best_idx]
        self.__DebugInfo = {
            "input_name": self.__InputName,
            "input_type": self.__InputType,
            "input_shape": self.__InputShape,
            "output_name": self.__OutputName,
            "selected_preprocess": best_result[3]["preprocess_name"],
            "selected_scale": best_result[3]["selected_scale"],
            "selected_margin": best_result[3]["margin"],
            "candidate_preprocesses": [
                {
                    "name": result[3]["preprocess_name"],
                    "confidence": float(np.max(result[1])),
                    "margin": result[3]["margin"],
                    "prediction": int(result[0]),
                    "selected_scale": result[3]["selected_scale"],
                    "preprocess_min": result[3]["preprocess_min"],
                    "preprocess_max": result[3]["preprocess_max"],
                    "preprocess_mean": result[3]["preprocess_mean"],
                    "scale_candidates": result[3]["scale_candidates"],
                    "raw_scores": result[3]["raw_scores"],
                }
                for result in results
            ],
            "selected_raw_scores": best_result[3]["raw_scores"],
            "selected_confidence": best_result[3]["confidence"],
        }
        return best_result[:3]

    def getSession(self) -> ort.InferenceSession:
        """내부 ONNX InferenceSession 객체를 반환합니다."""
        return self.__Session

    def getDebugInfo(self) -> dict:
        """가장 최근 추론의 디버그 정보를 반환합니다."""
        return self.__DebugInfo
