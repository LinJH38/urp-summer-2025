# Deep Learning for UAV Detection and Mode Classification (Anti-Drone)

**DroneRF** 데이터셋을 활용하여 무선 통신 RF 신호의 시계열 패턴을 바탕으로 미확인 드론의 존재 여부 및 비행 상태(Hovering, Flying 등)를 자동으로 분류하는 딥러닝 모델 구현 과정

## Project Overview
전장 및 일상 환경에서는 와이파이, 블루투스 등 다양한 무선 통신 간섭으로 인해 탐지하고자 하는 드론의 수신 RF 신호가 크게 훼손될 수 있음(배경 잡음, AWGN). 이 프로젝트는 노이즈가 섞인 원시 1D 시계열 RF 데이터를 입력받아 STFT 전처리와 2D CNN-LSTM을 거쳐 드론의 기종 및 비행 모드(예: DJI Phantom Video Recording 등 10개 클래스)를 예측하는 것.

## Dataset
* **DroneRF Dataset** (`10000L_1.csv` 등)
* **Features:** 2.4GHz 대역(Low/High band)의 시계열 데이터 및 2D 스펙트로그램 (Shape: `2 x 65 x 33`)
* **Labels:** 10개 비행 상태 (Background, Parrot AR Hovering, DJI Phantom Flying, Video Recording 등)
* **Environment:** 배경 잡음(Background Noise)만 있는 상태부터 드론의 과도 응답(Transient Response) 상태까지 포함

> **Note:** 데이터셋은 용량(약 43GB) 문제로 포함되어 있지 않음. 코드를 실행하려면 Mendeley Data에서 데이터를 다운로드하여 `data/raw/` 폴더에 위치.

## Model Architectures
1. **STFT Preprocessing:**
   * 1D 시계열 데이터를 2D 스펙트로그램으로 변환하여, 드론의 급격한 비행 모드 전환 시점(Time)과 주파수(Frequency) 특성을 동시에 확보.
2. **2D CNN + LSTM:**(여기서 구현x)
   * `Conv2d` 커널을 활용하여 특정 시간 프레임에서의 주파수 패턴(Feature) 추출.
   * `MaxPool2d`를 통해 시간 축(너비)은 보존하면서 주파수 축(높이)만 압축하여 일상적인 전파 노이즈를 억제.
   * 공간적/주파수적 특징은 CNN으로 추출하고, 그 특징들의 시계열적 흐름을 LSTM으로 기억하여 최종 분류를 수행합니다.

## Getting Started
```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 주피터 노트북 실행
jupyter notebook notebooks/01_DroneRF_Anti_Drone_tutorial.ipynb
```
