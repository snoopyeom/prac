### 2024년 한양대학교 인공지능1 기말 프로젝트

# YOLO

YOLO(You Only Look Once)는 실시간 객체 탐지를 위한 딥러닝 모델입니다. YOLO는 전체 이미지를 한 번에 분석하여 객체를 탐지하는 방식을 사용하며, 이는 전통적인 객체 탐지 방법들과는 달리 이미지를 여러 부분으로 나누지 않기 때문에 매우 빠르고 효율적입니다.
### YOLO의 주요 특징
*속도: 실시간 객체 탐지가 가능할 정도로 매우 빠르며, 이미지 전체를 한 번에 분석하기 때문에 프레임 당 수백 번의 분석 수행이 가능

*단일 신경망: 하나의 신경망이 이미지 전체를 분석하여 객체 위치와 클래스 확률을 동시에 예측

*End-to-End 학습: 전체 모델이 하나의 큰 신경망으로 구성되어 있어, 학습과정이 단순하고 직관적

*전역적 맥락 활용: 이미지 전체를 분석하기 때문에 객체의 전역적 맥락 정보를 잘 활용함

### YOLO의 작동 방식
1. 이미지 분할: 입력 이미지를 S x S 그리드 셀로 나눕니다.

2. 바운딩 박스와 확률 예측: 각 그리드 셀은 여러 개의 바운딩 박스(Bounding Box)와 각 바운딩 박스 내에 객체가 존재할 확률을 예측합니다. 각 바운딩 박스는 위치 (x, y), 크기 (width, height), 객체의 클래스 확률을 포함합니다.

3. 최대 신뢰도 바운딩 박스 선택: 각 그리드 셀에서 신뢰도가 가장 높은 바운딩 박스를 선택합니다.

4. Non-Maximum Suppression (NMS): 겹치는 바운딩 박스 중에서 가장 신뢰도가 높은 박스만 남기고 나머지를 제거하여 중복된 탐지를 없앱니다.

### YOLO의 버전
YOLO는 여러 버전이 있으며, 각 버전은 성능과 정확도가 개선되어 릴리즈되었습니다.

* YOLOv1: 최초의 YOLO 모델로, 이미지 분할과 바운딩 박스 예측의 기본 개념을 소개
  
* YOLOv2 (YOLO9000): 더 정확하고 빠르며, 여러 클래스에 대해 더 잘 일반화할 수 있도록 개선
  
* YOLOv3: 작은 객체 탐지 성능이 향상되었고, Darknet-53이라는 더 깊은 피처 추출 신경망을 사용
  
* YOLOv4: 다양한 최적화 기법을 통해 성능과 효율성을 더욱 높임

* YOLOv5: PyTorch로 구현된 YOLO 모델로, 사용의 편리성과 성능 측면에서 많이 사용됨
  
* YOLOv7, v8: 최신 버전으로, 다양한 새로운 기법들이 추가되어 성능 향상
  
* YOLOv9: Programmable Gradient Information(PGI)와 Generalized Efficient Layer Aggregation Network(GELAN)을 도입하여 데이터 손실 줄여 효율성 극대화
  
* YOLOv10: 실시간 객체 탐지 기능을 더욱 향상시켜 빠르고 정확한 객체 인식을 제공

YOLO는 다양한 응용 분야에서 널리 사용됩니다. 예를 들어, 자율 주행 차량에서 보행자나 다른 차량을 탐지하거나, 보안 카메라 시스템에서 침입자를 감지하는 데 사용되며, 모델의 빠른 속도와 높은 정확성 덕분에 실시간 응용 분야에서 특히 유용합니다.
 
 
# YOLOv5를 활용한 이미지 객체 탐지 프로젝트

## 분석개요

### 분석 배경
포장공정 마지막 단계인 X-Ray 검사기를 통한 완제품 내 금속 및 이물 포함 검출 과정에서 미검, 과검 문제 발견
### 분석 목표
Python을 이용해 주어진 X-Ray 검사 데이터셋 분석을 통해 장비셋팅, 검사기준 또는 여타 스펙을 개선할 수 있는 학습 모델 개발

![그림1](https://github.com/snoopyeom/prac/assets/19545380/859af51f-b47f-40e4-882a-b161d37a86cf)


## 제조 데이터 소개

### 데이터 수집 방법
* 제조 분야 : 분무건조공법을 이용한 분말유크림 제조
* 제조 공정명 : 포장공정 비전검사 단계
* 수집장비 : X-Ray 이물검출기의 SD Card
* 수집기간 : 2020/06/23 ~ 2020/09/22 (약 3개월)
* 수집조건 : 검사장비를 통과하는 제품이 불량일 경우에만 이미지를 저장

![그림2](https://github.com/snoopyeom/prac/assets/19545380/70943fe0-2762-491e-beea-ca73a2fdacfc)

### 데이터 크기, 데이터 수량
[X선이물검출기(06.23_09.22)] 폴더 내에 설비별로 [1호기], [2호기], [3호기] 폴더로 정리 되어 있으며, 설비별 폴더 안에는 각 일자별로 폴더가 정리되어 있고 일자별 폴더 안에 결함이미지(bmp)가 있음
[1호기] 폴더 – 1,031개, 112MB 
[2호기] 폴더 – 920개, 92.9MB 
[3호기] 폴더 – 858개, 210MB

### 불량품 데이터 설명
X-Ray에서 결함이 검출된 제품의 이미지는 다음과 같다. 붉은색 상자로 표시된 부분의 중심에 있는 검정색 부분이 제품의 결함 부분이며, 양품은 검정색 부분과 붉은색 상자로 표시된게 없는 데이터이며 따로 수집되지 않음

#### 독립변수 
* X-Ray 이미지 : 결함이 포함된 이미지 (BMP) 파일
* 결함 좌표 : 이미지 파일의 결함에 대해 좌표 정보가 담겨있는 TXT 파일

#### 종속변수
* 객체탐지 이미지 : 박스가 표시된 이미지 파일

---
## 모델선정

### YOLOv5
* 양품 이미지 데이터가 없고 오로지 결함 이미지 데이터만 갖고 있는 상황 = 객체 탐지 모델이 적합
* 객체 탐지를 위한 딥러닝 모델 중 가장 많이 쓰이고 있는 YOLO 채택
* YOLOv5는 상대적으로 최근에 출시된 버전으로, 경량화된 모델을 사용하여 빠른 추론 속도 제공하면서 객체 탐지의 정확성까지 유지

---

###  YOLO v5 학습 환경 셋팅
YOLOv5 모델을 학습하기 위해서는 환경 설정이 필요합니다.

#### 하드웨어 및 소프트웨어 요구 사항
GPU: CUDA를 지원하는 NVIDIA GPU가 필요합니다. YOLOv5는 GPU를 활용한 병렬 처리를 통해 학습 속도를 크게 향상시킬 수 있습니다.
CUDA 및 cuDNN: NVIDIA의 CUDA Toolkit과 cuDNN 라이브러리를 설치
Python: Python 3.6 이상 필요

#### 필수 라이브러리 설치
PyTorch: YOLOv5는 PyTorch 프레임워크를 사용합니다. PyTorch는 [PyTorch](https://pytorch.org/)에서 설치할 수 있습니다.

기타 라이브러리: 필요한 추가 Python 라이브러리들은 requirements.txt 파일에 명시되어 있습니다. 이를 설치하려면 다음 명령을 사용하면 된다.

    pip install -r requirements.txt

#### 데이터 준비
* 데이터셋 구조: 데이터는 train, val 디렉토리와 함께 클래스별로 구분된 이미지 파일들로 구성됩니다.
* 레이블 파일: 각 이미지에 대해 YOLO 형식의 레이블 파일이 필요하며, 레이블 파일은 각 객체에 대한 클래스 번호, 중심 좌표 (x, y), 폭과 높이 (width, height)를 포함합니다.

#### 환경 설정
* YOLOv5 클론: YOLOv5 코드를 GitHub에서 클론합니다:
```
    git clone https://github.com/ultralytics/yolov5
    cd yolov5
```

* Config 파일 수정: 학습 설정 파일을 필요에 맞게 배치 크기(batch size), 학습률(learning rate), 에포크 수(epochs) 등을 수정합니다.


---
### 데이터 배치

#### 디렉토리 구성 맞추기
* YOLOv5의 고유 폴더 배치 규칙
1. Yolov5-data-images 폴더에 독립변수 중 X-Ray 이미지와 labels 폴더에는 결함 좌표가 담긴 txt 파일을 넣어줍니다.
2. 이미지 파일과 txt 파일이 1:1로 대치되어야 합니다. (확장자명을 제외한 파일명이 같아야 함)


#### Labelme를 통한 라벨링 작업
Output Directory를 Yolov5-data-labels 로 맞춰주고 라벨링 작업 진행합니다.
Labelme를 사용하면 json 파일로 결과물이 저장되어 추후 yolo 형식에 맞는 txt 파일로 변환 시켜줘야 합니다.

< labelme 설치법 및 사용법 >
Labelme는 이미지에 라벨을 달기 위한 오픈 소스 도구로, 주로 객체 탐지와 이미지 분할을 위한 데이터셋 준비에 사용됩니다. Labelme 설치 방법은 다음과 같습니다:

1. Python 및 pip 설치
Labelme는 Python 기반의 도구이므로 Python이 설치되어 있어야 합니다. [Python 설치하기](https://www.python.org/downloads/)

2. Labelme 설치
Labelme는 pip를 사용하여 설치할 수 있습니다. 터미널이나 명령 프롬프트를 열고 다음 명령어를 입력하세요.
```
    pip install labelme
```
3. 설치 확인
설치가 완료되면 labelme 명령어를 입력하여 Labelme가 제대로 설치되었는지 확인합니다.
```
    labelme
```

#### yaml 파일 생성
위에서 맞춰준 데이터셋 정보를 담고 있는 파일을 생성해줘야 합니다.
Yolov5-data 경로에 data.yaml 생성

![image](https://github.com/snoopyeom/prac/assets/19545380/6be86b5a-d4cc-4a9a-9f45-c3747a079612)
* train, val 의 경로는 추후 코드 통해 수정
* nc : 클래스 개수 
* names : 클래스 이름 지정 

Bmp로 구성되어 있는 이미지들을 모두 YOLOv5 형식에 맞게 jpg로 변환해준다
디렉토리 내 모든 파일의 확장자를 jpg로 변경

Json으로 구성되어 있는 라벨링 파일들을 모두 YOLOv5 형식에 맞게 txt파일로 변환해준다
좌표값도 YOLOv5형식에 맞게 변환


---
### 훈련 모델, 학습 모델 생성
Images 폴더 안에 모든 이미지들을 list로 만든다

* 경로, 패턴을 사용하여 파일들을 검색하는 glob 함수 활용한다.
![image](https://github.com/snoopyeom/prac/assets/19545380/12f80562-2055-4139-8a2e-14ef7ef86a9f)

* 해당 img_list를 train set과 valid set 8:2로 나눠준다.
![image](https://github.com/snoopyeom/prac/assets/19545380/2d0480d7-d919-4dea-bd2f-09e855c929d7)

* 나눠준 train, valid set을 각각 txt파일로 만들어준다. 이미지 주소들이 txt 파일에 write 된다.

![image](https://github.com/snoopyeom/prac/assets/19545380/589a8e1b-607d-451b-8659-cf3bebd70a5e)  ![image](https://github.com/snoopyeom/prac/assets/19545380/f0d8d59e-ecaa-4403-ae3a-5ef8060232a0)

* 앞서 만들어준 data.yaml 파일의 train, val 경로를 재설정해준다.
![image](https://github.com/snoopyeom/prac/assets/19545380/dfa45622-2757-4d9f-b530-d17cba9f12ef)

---
### 모델 학습
모델 학습 하이퍼파라미터
Img : 이미지 크기 정하는 옵션
Batch : 역전파 단계에 사용되는 이미지 샘플의 수 Epochs : 전체 데이터셋을 한 번 훈련하는 단위 
Data : 데이터 구성 파일의 경로
Cfg : YOLOv5 모델 구성 파일의 경로
Weights : 사전 훈련된 가중치 파일의 경로 
Name : 훈련 결과 저장할 디렉토리 이름 지정. 훈련 중 생성되는 가중치 파일, 로그 파일 및 결과 이미지 등이 저장됨
![image](https://github.com/snoopyeom/prac/assets/19545380/0339c78d-f6fb-4758-9022-f6c77fdb9ea1)
훈련 결과가 runs/train/gun_yolov5s_results8에 저장됨 (추후 확인)
![image](https://github.com/snoopyeom/prac/assets/19545380/076adda7-ee65-43cf-8640-df902cd4f637)
추론에 사용할 수 있는 last.pt(최종 weight), best.pt(가장 성능이 좋았을 때 weight)파일이 weights 파일에 저장됨 (추후 사용) 
![image](https://github.com/snoopyeom/prac/assets/19545380/3aa6ae5b-cfd0-4b5b-ac7f-b9528569e1d0)


---
### 훈련 결과
#### Yolov5/runs/train/gun_yolov5s_results8폴더를 통해 여러 결과 값을 확인
* Precison : 모델이 올바르게 예측한 비율 (모델이 예측한 결과 중 실제 양품인 비율)
* Recall : 모델이 실제로 발견해야 하는 것 중 얼마나 발견했는지 비율 (실제로 양품인 것 중 모델이 올바르게 예측한 비율)
* mAP : Precision과 Recall을 종합적으로 평가하는 지표

![image](https://github.com/snoopyeom/prac/assets/19545380/413eb4ee-51d3-4dbc-8c3f-9721a348257d)

![image](https://github.com/snoopyeom/prac/assets/19545380/a42505a5-4fa4-4bd4-a337-ec6053736968)

에폭이 진행됨에 따라 3가지 성능지표 모두 1과 가까워지는 것을 확인할 수 있습니다.
또한 학습 과정에서 validation 데이터를 테스트 한 결과를 이미지로 볼 수 있습니다.


---
### 모델 활용
####앞서 저장된 best.pt 파일을 통해 성능을 검증 진행

* Conf : 신뢰도 값. 0과 1사이의 숫자로, 값을 높이면 더 확실한 객체만 검출됩니다.

![image](https://github.com/snoopyeom/prac/assets/19545380/8095387b-09b6-4bb2-b3bc-f444bc85144b)
![image](https://github.com/snoopyeom/prac/assets/19545380/6e8428d5-d975-48e2-801a-d0397e82818f)

* runs/detect/exp에 저장된 결과값 확인

![image](https://github.com/snoopyeom/prac/assets/19545380/96356f15-dcdd-44b8-a607-e4875b3dd86b)


---
### 모델 활용 - 오토라벨링
생성된 활용해 기존 이미지에 대한 오토라벨링을 진행합니다.

* Source : 라벨링 할 이미지가 위치한 폴더
* Weight : 사전 훈련된 모델 가중치 파일의 경로
* Save : 저장 방식
* Exist : 디렉토리가 이미 존재하는 경우 오류 발생시키지 않고 계속 진행
* Project : 출력될 폴더 

![image](https://github.com/snoopyeom/prac/assets/19545380/7dbc8b21-1f3e-4347-b602-52c0c5041859)
![image](https://github.com/snoopyeom/prac/assets/19545380/c2b689cf-1127-411d-bfec-e2d70e183064)


* Yolov5/data/unit_1_label에 저장된 값 확인

![image](https://github.com/snoopyeom/prac/assets/19545380/0fa43f22-d44b-4715-b31e-ac7e255170ed)

![image](https://github.com/snoopyeom/prac/assets/19545380/44977f4e-6ffe-4eb1-8a2c-93ac9b72820c)

![image](https://github.com/snoopyeom/prac/assets/19545380/1796b040-1c60-4122-b42c-0dc776570a04)




  





















