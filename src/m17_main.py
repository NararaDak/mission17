import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import os
import json
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from m17_model import MnistModel

APP_VERSION = "0004"

# ── 경로 설정 ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "data", "modelfiles", "mnist-12-int8.onnx")

# ── 모델 로드 (캐싱) ──────────────────────────────────────────────────────────
@st.cache_resource
def Load_Model(cache_key: str) -> MnistModel:
    return MnistModel(MODEL_PATH)


def Get_Debug_Text(debug_info: dict) -> str:
    return json.dumps(debug_info, ensure_ascii=False, indent=2)

# ── 페이지 설정 ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="MNIST 숫자 인식", page_icon="✏️", layout="centered")

st.title("✏️ 손글씨 숫자 인식기")
st.markdown(
    "아래 캔버스에 **0~9 숫자**를 마우스로 그린 뒤 결과를 확인하세요.\n\n"
    "- 검정 배경에 하얀 선으로 그립니다.\n"
    "- 그림 도구·선 굵기를 사이드바에서 조절할 수 있습니다.\n"
    f" ver {APP_VERSION}"
)

if st.session_state.get("app_version") != APP_VERSION:
    st.session_state["app_version"] = APP_VERSION
    st.session_state.pop("prediction_result", None)

# ── 사이드바 설정 ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    stroke_width = st.slider("선 굵기", min_value=5, max_value=40, value=18, step=1)

# ── 캔버스 ────────────────────────────────────────────────────────────────────
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=stroke_width,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    update_streamlit=False,
    key="canvas",
)

predict_clicked = st.button("🔍 예측하기", use_container_width=True)

# ── 예측 ──────────────────────────────────────────────────────────────────────
if predict_clicked:
    if canvas_result.image_data is None:
        st.info("캔버스에 숫자를 그려 주세요.")
        st.stop()

    # RGBA → PIL
    img_rgba = Image.fromarray(canvas_result.image_data.astype(np.uint8), "RGBA")

    # 배경(검정)만 있으면 아직 아무것도 그리지 않은 상태로 판단
    gray = np.array(img_rgba.convert("L"), dtype=np.uint8)
    if gray.max() < 10:
        st.info("캔버스에 숫자를 그려 주세요.")
    else:
        model = Load_Model(APP_VERSION)
        pred_class, probs, arr28 = model.doPredict(img_rgba)
        debug_info = model.getDebugInfo()
        st.session_state["prediction_result"] = {
            "pred_class": int(pred_class),
            "probs": probs.tolist(),
            "arr28": arr28,
            "debug_info": debug_info,
        }

result = st.session_state.get("prediction_result")
if result is not None:
    pred_class = result["pred_class"]
    probs = np.array(result["probs"], dtype=np.float32)
    arr28 = result["arr28"]
    debug_info = result["debug_info"]
    debug_text = Get_Debug_Text(debug_info)

    st.markdown(f"## 예측 결과: **{pred_class}**")
    st.progress(float(probs[pred_class]), text=f"신뢰도: {probs[pred_class]*100:.1f}%")

    with st.expander("전체 클래스 확률 보기"):
        prob_data = {str(i): float(f"{probs[i]*100:.2f}") for i in range(10)}
        st.bar_chart(prob_data, x_label="숫자", y_label="확률 (%)")

    with st.expander("모델 입력 이미지(28×28) 미리보기"):
        preview = Image.fromarray((arr28 * 255).astype(np.uint8)).resize(
            (140, 140), Image.NEAREST
        )
        st.image(preview, caption="전처리 후 모델 입력", clamp=True)

    if st.button("디버그내용복사", key="copy_debug_button"):
        safe_debug_text = json.dumps(debug_text)
        components.html(
            f"""
            <script>
            const debugText = {safe_debug_text};
            navigator.clipboard.writeText(debugText).then(() => {{
                window.parent.postMessage({{type: 'streamlit:setComponentValue', value: 'copied'}}, '*');
            }}).catch(() => {{
                window.parent.postMessage({{type: 'streamlit:setComponentValue', value: 'failed'}}, '*');
            }});
            </script>
            """,
            height=0,
        )
        st.success("디버그 메시지를 클립보드에 복사했습니다.")

    with st.expander("디버그 패널"):
        st.write(
            {
                "input_name": debug_info.get("input_name"),
                "input_type": debug_info.get("input_type"),
                "input_shape": debug_info.get("input_shape"),
                "output_name": debug_info.get("output_name"),
                "selected_preprocess": debug_info.get("selected_preprocess"),
                "selected_scale": debug_info.get("selected_scale"),
                "selected_confidence": debug_info.get("selected_confidence"),
                "selected_margin": debug_info.get("selected_margin"),
            }
        )

        st.write("선택된 raw scores")
        st.write(debug_info.get("selected_raw_scores", []))

        st.write("전처리 후보별 결과")
        for candidate in debug_info.get("candidate_preprocesses", []):
            st.write(
                {
                    "name": candidate.get("name"),
                    "prediction": candidate.get("prediction"),
                    "confidence": candidate.get("confidence"),
                    "margin": candidate.get("margin"),
                    "selected_scale": candidate.get("selected_scale"),
                    "preprocess_min": candidate.get("preprocess_min"),
                    "preprocess_max": candidate.get("preprocess_max"),
                    "preprocess_mean": candidate.get("preprocess_mean"),
                }
            )
            st.write("scale candidates")
            st.write(candidate.get("scale_candidates", []))
            st.write("raw scores")
            st.write(candidate.get("raw_scores", []))
else:
    st.info("숫자를 그린 뒤 '예측하기' 버튼을 눌러 주세요.")
