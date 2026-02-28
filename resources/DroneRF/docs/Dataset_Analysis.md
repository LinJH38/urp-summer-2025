# DroneRF Dataset Analysis (Anti-Drone RF Time-Series)

이 문서는 불법 드론 및 무인기 탐지를 위한 딥러닝 모델을 설계하기 위해 DroneRF 데이터셋의 특성과 2.4GHz 무선 통신 환경의 물리적/수학적 배경을 정리한 노트입니다.

## 1. 데이터셋 구조 (Data Shape & Hierarchy of Classes)
* **Shape:** `[Total Windows, 2048(Sequence Length), 2(Low/High Bands)]` (원시 시계열 데이터를 AI 입력에 맞게 $2048$ 단위로 윈도잉하여 분할)
* **Classes (10 Flight Modes):** 문제의 난이도에 따라 3단계의 계층적(Hierarchical) 구조를 가짐.
  * **Level 1 (드론 유무 탐지):** Background (드론 없음) vs. Active Drone
  * **Level 2 (드론 기종 식별):** Parrot AR, Parrot Bebop, DJI Phantom
  * **Level 3 (비행 상태 세부 인지):** * On and Connected (전원 인가 및 조종기 연결)
    * Hovering (제자리 비행)
    * Flying (이동 비행 중)
    * Video Recording (비디오 촬영 및 지상 전송 중 - 데이터 링크의 트래픽 폭증으로 파동 밀도가 급변함)

## 2. 2.4GHz 주파수 대역 분할과 RF 신호의 이해 (Signal Characteristics)
* 드론과 지상 조종기(Controller)는 주로 $2.4\text{GHz}$ ISM 대역을 통해 제어 명령과 텔레메트리/비디오 데이터를 교환함.
* 수신 안테나를 통해 들어오는 광대역(Wideband) 주파수 스펙트럼을 처리의 편의성과 해상도를 위해 기저대역(Baseband)으로 변환 후, 스펙트럼을 절반으로 쪼개어 수집함:
  * **Low-band:** $2.400\text{GHz} \sim 2.440\text{GHz}$
  * **High-band:** $2.440\text{GHz} \sim 2.480\text{GHz}$



* 딥러닝 모델(CNN-LSTM)은 이 2개 채널의 시간 축 진폭(Amplitude) 변화를 입력받아, 특정 드론 제조사 고유의 주파수 호핑(FHSS, Frequency Hopping Spread Spectrum) 패턴을 추출해 냄.

## 3. 전장 노이즈 환경과 STFT (Time-Frequency Analysis)
* **Background 잡음 모델링:** 드론이 없는 상태(Class 0)는 완전한 무음이 아니라 Wi-Fi, 블루투스 등 일상적인 전파가 혼재된 상태임. 확률변수론적 관점에서 이는 순수한 가산 백색 가우시안 잡음(AWGN) $X \sim \mathcal{N}(\mu, \sigma^2)$ 뿐만 아니라 복잡한 유색 잡음(Colored Noise)이 섞인 랜덤 프로세스(Random Process)로 묘사됨.
* **STFT 전처리:** 단순 1D 시계열로는 드론의 급격한 기동 변화(과도 응답, Transient Response)를 포착하기 어려움. 따라서 단기 푸리에 변환(STFT)을 거쳐 시간-주파수 2D 스펙트로그램(Spectrogram)으로 변환, 비행 모드 전환 시점의 해상도를 극대화함.

## 4. 실무적 응용성 (Applicability): 대드론 방공망 (Anti-Drone System)
* **수동형 RF 인터셉트 및 조기 경보:** 시야에 보이지 않거나 기존 SAR 레이더의 반사 면적(RCS)이 너무 작아 잡히지 않는 초소형 드론이 접근할 때, $2.4\text{GHz}$ 조종 RF 신호만 수동형(Passive)으로 가로채어 즉각적인 드론의 기동 상태를 파악함.



* **엣지 디바이스 및 보병 작전 연계:** 무거운 서버가 아닌 ARM 코어 기반의 마이크로프로세서(SoC)에 경량화된 AI 모델을 탑재함. 이를 통해 전장의 보병이 휴대하는 안티 드론 건(Anti-Drone Gun)이나 소형 초소 레이더에 실시간 알림을 제공하고, 파악된 통신 방식을 역이용해 타겟 맞춤형 재밍(Jamming)을 수행할 수 있도록 대비함.
