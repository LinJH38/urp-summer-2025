# RadioML 2018.01a Dataset Analysis(tutorial dataset은 용량 문제로, 2016.10a 활용)

이 문서는 자동 변조 분류(AMC) 모델을 설계하기 위해 RadioML 데이터셋의 특성과 무선 통신 환경의 물리적/수학적 배경을 정리한 노트입니다.

## 1. 데이터셋 구조 (Data Shape & Classes)
* **Shape:** [255904(Total Frames), 1024(Sequence Length), 2(I/Q Channels)]
* **Classes (24 Modulation Schemes):**
  * 아날로그 변조: AM-DSB, AM-SSB, WBFM, ...
  * 디지털 변조: BPSK, QPSK, 8PSK, 16QAM, 64QAM, CPFSK, GFSK, PAM4, ...

## 2. I/Q 신호와 고주파 신호의 이해 (Signal Characteristics)
* 안테나를 통해 수신되는 고주파(High-frequency) 통신 신호를 처리의 편의성을 위해 기저대역(Baseband)으로 Down-conversion
* 이 과정에서 신호는 동상(In-phase, $I$) 성분과 직교(Quadrature, $Q$) 성분으로 나뉘어 복소수 형태의 이산 시간 신호(Discrete-time signal)로 표현:
  $s[t] = I(t)cos(2pi*f_c*t) - Q(t)sin(2pi*f_c*t)& -> $x[t] = I[t] + jQ[t]$ f_c(반송파 주파수) 제거
* CNN과 LSTM은 이 두 채널의 위상 변화(Phase Shift)와 진폭 패턴을 학습하여 변조 방식을 역추적

## 3. 노이즈 환경과 SNR (Signal-to-Noise Ratio)
* 무선 채널을 통과한 신호는 가산 백색 가우시안 잡음(AWGN, Additive White Gaussian Noise)의 영향을 받음 
* 확률변수론적 관점에서 AWGN은 평균이 0이고 분산이 $\sigma^2$인 정규분포 $\mathcal{N}(0, \sigma^2)$를 따름
* **SNR Range:** $-20\text{dB}$ 부터 $+18\text{dB}$ 까지 존재하며, SNR이 낮을수록 노이즈의 전력이 신호의 전력보다 압도적으로 커져 분류 난이도가 상승

## 4. 실무적 응용성 (Applicability): AMC(Automatic Modulation Classification)
* **인지 무선(Cognitive Radio):** 주변 주파수 환경을 감지하고 빈 주파수 대역을 찾아 효율적으로 통신하는 기술에 필수적
* **국방 및 전자전(Electronic Warfare):** 적군의 통신 신호를 감청하고, 어떤 변조 방식을 사용하는지 실시간으로 파악하여 재밍(Jamming)을 수행하거나 아군의 통신 보안 강화에 활용
