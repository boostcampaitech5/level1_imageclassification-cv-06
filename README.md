# Mask Image Classification

## Project Overview
### 프로젝트 목표
 - 임의의 사진이 주어졌을 때, **나이, 성별, 마스크 착용 여부**를 판단하는 모델 제작

### 기대 효과
- 해당 프로젝트를 이용한 시스템을 통해 적은 비용으로 마스크 정상 착용 여부를 판별할 수 있을 것으로 기대

### Dataset
- 20 ~ 70대 아시아인 4,500명에 대한 사진 7장( 착용 (5 장) , 잘못된 착용 (1 장) , 미착용 (1 장) )
- 해상도 : 384, 512
- 총 이미지 수 : 31,500장(Train 이미지 수 : 18,900장)

### GPU
- V100(vram 32GB) 5개

### 평가기준
- F1-score

<br>
  
## Team Introduction
### Members
| 고금강 | 문상인 | 박재민 | 박종서 | 주재영 |
|:--:|:--:|:--:|:--:|:--:|
|<img  src='https://avatars.githubusercontent.com/u/101968683?v=4'  height=80  width=80px></img>|<img  src='https://avatars.githubusercontent.com/u/78636931?v=4'  height=80  width=80px></img>|<img  src='https://avatars.githubusercontent.com/u/68144124?v=4'  height=80  width=80px></img>|<img  src='https://avatars.githubusercontent.com/u/59987079?v=4'  height=80  width=80px></img>|<img  src='https://avatars.githubusercontent.com/u/103994779?v=4'  height=80  width=80px></img>|
|[Github](https://github.com/TwinKay)|[Github](https://github.com/moons98)|[Github](https://github.com/jemin7709)|[Github](https://github.com/justinpark820)|[Github](https://github.com/JaiyoungJoo)|
|twinkay@yonsei.ac.kr|munsi2003@gmail.com|jemin7709@gmail.com|justin93820@gmail.com|wodud3851@gmail.com|

### Members' Role

| 팀원 | 역할 |
| -- | -- |
| 고금강_T5011 | EDA를 통한 data re-labeling 진행, 실험 수행 |
| 문상인_T5075 | 자동화를 위한 baseline 코드 구현, 실험 설계 및 분석 레포트 작성 |
| 박재민_T5089 | 자동화를 위한 baseline, shell script code 구현, Loss 및 TTA 구현 |
| 박종서_T5092 | 다양한 모델, Loss, Ensemble 코드 구현 및 검증 |
| 주재영_T5207 | 실험 수행 및 결과 로깅 및 관리 |


### Our Notion

<a  href="https://www.notion.so/Hot-6-fd88defc7268499386e8179658d173f3"><img  src="https://upload.wikimedia.org/wikipedia/commons/4/45/Notion_app_logo.png"  height=100  width=100px/></a>

<br>

## Procedure & Techniques

  

| 분류 | 내용 |
| -- | -- |
|Dataset|**Mis-labeled data** <br> - 전수 조사를 통한 Mis-labeled data 발견 및 relabeling <br> <br>  **Class imbalance data** <br> - 마스크 착용 여부에 따른 데이터 불균형 (5:1:1) <br> - 성별에 따른 데이터 불균형 (약 1:1.6) <br> - 나이에 따른 데이터 불균형 (약 7:6:1) <br> &nbsp;&nbsp;&nbsp;&nbsp;=> age band 수정 <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=>class0 < 30, 30<= class1 < 59, 59<= class2 <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=>class0 < 30, 30<= class1 < 58, 58<= class2 <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;=>class0 < 29, 29<= class1 < 58, 58<= class2 <br> <br> **BackGround Subtraction** <br> - Dataset의 일부에서, 배경에 사람 얼굴이 존재함을 발견하고 '[rembg](https://github.com/danielgatis/rembg)'라는 오픈소스를 활용하여 배경 제거 
|Augmentation|**CenterCrop** <br> - Female, 18~19세 상당수의 인원이 특정 클래스로 특정되는 옷(빨간색 트레이닝 복, 군복)을 입은 것을 확인 <br> - centercrop을 사용하여 옷 정보를 최대한 제거 <br> &nbsp;&nbsp;&nbsp;&nbsp;=> CenterCrop(360,360)에서 가장 높은 성능을 보임  <br> <br> **RandomHorizontalFlip** <br> **RandomRotation** <br> **RandomHorizontalFlip** <br> **ColorJitter** <br> **ToTensor** <br> **Normalize**
|Loss|**Focal Loss** <br> - Class 불균형 문제가 있어 사용, 확실한 성능 향상 존재 <br> <br>  **Label Smoothing Loss** <br> - 초기 빠른 수렴과 overfitting 방지를 위해 사용, relabel 이후 다른 Loss들과 같이 사용했을 때 성능 향상 존재<br>  <br> **F1 Loss** <br> - 평가지표가 F1 Score이었기 때문에 성능 향상이 있을 것이라 생각하여 사용 (1 - F1 Score)으로 구현 <br>  <br> &nbsp;&nbsp;&nbsp;&nbsp;=> 최종적으로 **Focal Loss**, **Label Smoothing Loss**, **F1 Loss** 모두 사용 (동일 가중치)
|Model|**Resnet50** <br> - Resnet50을 우선적으로 선택하고 Augmentation 실험을 진행하여 최적화 시켰으나, 타 모델에서는 같은 세팅에서 Resnet보다 낮은 성능을 보임 <br> &nbsp;&nbsp;&nbsp;&nbsp;=> 시간 부족의 이유로 Resnet을 그대로 사용하기로 결정
|Fast Training skills|**Mixed Precision(AMP)** <br> - 좀 더 빠른 학습과 최적화를 위해 사용 <br> &nbsp;&nbsp;&nbsp;&nbsp;=> 기존보다 많으면 절반 수준까지 메모리를 덜 사용하는 모습 확인 <br> <br>  **Scheduler** <br> - 기존 Cosine Annealing with Warm Restart(이하 CAWR)는 max값이 고정되어있기 때문에 기존에 사용 중인 StepLR대비 후반부의 안정적인 학습이 힘들 것이라 생각하여 사이클에 따라 max값을 줄여나가는 Custom CAWR 스케줄러를 사용 <br> &nbsp;&nbsp;&nbsp;&nbsp;=> 기존 대비 학습 종료 epoch이 절반으로 줄었고 학습 수렴 자체는 10배정도 빠르게 실험 가능
|Ensemble & TTA|**Ensemble** <br> - soft voting을 선택하여 구현 (softmax 사용) <br> - age band 수정 1,3번 데이터, seed : 42, random 총 Resnet50 4개로 앙상블 진행 <br>  <br>  **TTA** <br> - 모델의 틀린 예측을 최대한 방지하기 위해 TTA를 적용 <br> - 5회 진행|

  
## Training History

<img  src='https://i.esdrop.com/d/f/JnUd4EePHi/ufrcOZsRn4.png'  height=500  width=900px></img>

<br>

## Getting Started
### install requirements

```
pip install -r requirements.txt
```

<br>

### Train (Auto Training)
./configs/queue 에 원하는 실험들의 config.json 파일 생성(base_config 이용) 후, <br>
<img  src='https://i.esdrop.com/d/f/JnUd4EePHi/QMc7YMKBAE.png'  height=300  width=200px></img>
<br>
```
sh auto_trainer.sh
```
### Inference
```
python inference.py
```
