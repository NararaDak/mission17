# Mission17 개발 규칙 문서

## 1. 문서 역할

이 문서는 Mission17 프로젝트의 개발 규칙 문서입니다.

포함 범위:

- Python 코드 작성 규칙
- 네이밍 규칙
- 변수 범위별 이름 규칙
- 함수 / 클래스 / 멤버 설계 기준

포함하지 않는 범위:

- 전체 시스템 구조
- 추론 파이프라인 설명
- 기술 스택 설명
- 상세 로직 다이어그램

위 내용은 TECH.md에서 관리합니다.


## 2. 기본 원칙

1) 한 프로젝트 안에서는 네이밍 스타일을 섞지 않는다.

2) 이 프로젝트는 다음 스타일을 기준으로 한다.

- 상수: 대문자 + `_`
- 전역 함수: 대문자 시작 + `_`
- 클래스: PascalCase
- 공개 메서드: `get/set/do + CamelCase`
- 내부 메서드: `_name`
- 파라미터/지역 변수: 소문자 시작 + 대문자 구분
- 내부 멤버 변수: `self.__Name`
- 외부 공개 멤버 변수: `self._Name`

3) 의미가 같은 값은 범위가 달라도 핵심 단어를 유지한다.


## 3. 상수 규칙

상수는 모두 대문자로 쓰고, 의미 구분은 `_` 로 한다.

예:

```python
EPOCHS = 100
CAT_NUMBER = 10
DOG_NUMBER = 10
MODEL_PATH = "..."
APP_VERSION = "0004"
```


## 4. 전역 함수 규칙

전역 함수는 대문자로 시작하고, 단어 구분은 `_` 를 사용한다.

예:

```python
def Load_Model():
	pass

def Get_Debug_Text():
	pass

def Build_Result_Data():
	pass
```

전역 함수는 앱 전체에서 공통으로 쓰는 기능만 둔다.


## 5. 클래스 규칙

클래스 이름은 PascalCase를 사용한다.

예:

```python
class CleanData:
	pass

class GetLoader:
	pass

class MnistModel:
	pass
```


## 6. 멤버 변수 규칙

### 6-1. 내부 전용 멤버 변수

기본적으로 클래스 변수는 외부에 공개하지 않는 것을 원칙으로 한다.

형식:

```python
self.__Session
self.__InputShape
self.__DebugInfo
```

의미:

- 클래스 내부에서만 직접 사용
- 외부에서 바로 접근하지 않음


### 6-2. 외부 공개 멤버 변수

정말 필요한 경우에만 다음 형태를 사용한다.

```python
self._ModelName
```

이 프로젝트에서는 가능한 한 getter/setter 또는 공개 메서드로 접근한다.


## 7. 멤버 함수 규칙

### 7-1. 내부 전용 함수

클래스 내부에서만 쓰는 함수는 `_` 로 시작한다.

예:

```python
def _process(self):
	pass

def _preprocess(self):
	pass
```


### 7-2. 외부 공개 함수

외부에 노출하는 함수는 다음 규칙을 사용한다.

- `getMyName()`
- `setMyName(name)`
- `doJob()`

원칙:

- `get`, `set`, `do` 로 시작
- 그 다음 단어는 대문자 시작
- `_` 는 사용하지 않음

예:

```python
def getMyName(self):
	pass

def setMyName(self, name):
	pass

def doPredict(self, inputImage):
	pass
```


## 8. 파라미터 변수명 규칙

파라미터 변수명은 소문자로 시작하고, 의미 구분은 대문자로 한다.

예:

```python
def doJob(filePath, modelPath, inputImage):
	pass
```

원칙:

- 약어보다 의미가 분명한 이름을 사용
- 너무 짧은 이름은 피함
- 범위보다 역할이 드러나는 이름을 우선함

좋은 예:

- `filePath`
- `modelPath`
- `inputImage`
- `debugInfo`

나쁜 예:

- `x`
- `img`
- `n`
- `file_Path`


## 9. 지역 변수명 규칙

지역 변수도 소문자로 시작하고, 의미 구분은 대문자로 한다.

예:

```python
filePath = "a.txt"
modelInput = None
predClass = 3
rawScores = []
debugText = "..."
```

원칙:

- 함수 안에서만 쓰더라도 의미가 보이게 작성
- 파라미터와 이름 충돌이 나지 않게 작성
- 가능하면 파라미터와 핵심 단어를 공유


## 10. 변수 범위별 이름 구분 규칙

전역 상수:

- `MODEL_PATH`
- `APP_VERSION`

전역 함수:

- `Load_Model()`
- `Get_Debug_Text()`

파라미터 변수:

- `modelPath`
- `inputImage`
- `debugInfo`

지역 변수:

- `predClass`
- `rawScores`
- `debugText`

클래스 내부 변수:

- `self.__Session`
- `self.__InputShape`

외부 공개 멤버 변수:

- `self._ModelName`


## 11. 이름 일관성 규칙

같은 의미의 이름은 범위가 달라도 가능한 한 같은 단어를 유지한다.

예:

- 파라미터: `modelPath`
- 지역 변수: `currentModelPath`, `savedModelPath`

- 파라미터: `inputImage`
- 지역 변수: `grayImage`, `resizedImage`

이름이 완전히 달라지면 추적이 어려우므로,
같은 개념은 같은 핵심 단어를 반복해서 사용한다.


## 12. 예외 규칙

캐시 키, 세션 상태 키, JSON 딕셔너리 키처럼 외부 시스템과 주고받는 문자열 값은
가독성과 호환성을 위해 소문자 또는 기존 포맷을 유지할 수 있다.

예:

- `"prediction_result"`
- `"input_name"`
- `"selected_scale"`


## 13. 문서 관리 원칙

이 문서에는 개발 규칙만 유지한다.

기술 설명, 파이프라인, 로직 흐름, 디버그 구조, 기술 스택은 TECH.md에 기록한다.

