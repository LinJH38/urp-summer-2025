# Variational Autoencoder (VAE)

## 1. Problem Scenario
Continuous latent variables $z$를 포함하는 directed probabilistic models에서 inference와 learning 수행법

### 1.1 기본 설정
- $N$개의 i.i.d. dataset $X$를 지님.
- **데이터 생성 과정의 가정**
    1. $z$가 prior $p_{\theta}(z)$에 의해 생성
    2. $x$가 conditional distribution $p_{\theta}(x|z)$에 의해 생성

### 1.2 Intractability
$$p_{\theta}(x) = \int p_{\theta}(x|z)p_{\theta}(z)dz$$
- 위 식을 최대화해야 하지만, $p_{\theta}(x|z)$는 Deep Neural Network의 non-linearity를 지니므로 매우 복잡함 (Not closed solution form). 따라서 적분 자체가 **intractable**함.
- Posterior $p_{\theta}(z|x) = p_{\theta}(x|z)p_{\theta}(z)/p_{\theta}(x)$ 역시 $p_{\theta}(x)$로 인해 intractable함.

---

## 2. Solution: Variational Lower Bound
True posterior $p_{\theta}(z|x)$의 approximation인 $q_{\phi}(z|x)$를 도입하여, log-likelihood $\log p_{\theta}(x)$를 다음과 같이 정리 가능함.

$$\log p_{\theta}(x) = D_{KL}(q_{\phi}(z|x) \| p_{\theta}(z|x)) + \mathcal{L}(\theta, \phi; x)$$

- $\mathcal{L}$은 Lower bound이며 **ELBO (Evidence Lower Bound)**로 부름.
- 첫 번째 항인 KL Divergence는 $p_{\theta}(z|x)$가 intractable하기 때문에 control 불가능함 (단, $D_{KL} \geq 0$).
- 따라서, Lower bound($\mathcal{L}$)를 Maximize(MLE) 시킴으로써 간접적으로 objective인 $\log p_{\theta}(x)$를 Maximize 시킬 수 있음.

### 2.1 ELBO (Evidence Lower Bound)
$$\mathcal{L}(\theta, \phi; x) = -D_{KL}(q_{\phi}(z|x) \| p_{\theta}(z)) + \mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(x|z)]$$

식은 **Regularization Term**과 **Reconstruction Term**의 형태로 구성됨.

1.  **Regularization Term**: $-D_{KL}(q_{\phi}(z|x) \| p_{\theta}(z))$
    - Prior $p_{\theta}(z) \sim N(0, I)$와 Posterior approximation $q_{\phi}(z|x) \sim N(\mu, \sigma^2)$를 최대한 가깝게 만들어야 함.
    - Analytically, 다음과 같이 계산됨:
      $$\frac{1}{2} \sum_{j=1}^{J} (1 + \log((\sigma_j)^2) - (\mu_j)^2 - (\sigma_j)^2)$$
2.  **Reconstruction Term**: $\mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(x|z)]$
    - 인코더에서 나온 $z$를 통해 디코더가 얼마나 원본 데이터 $x$를 잘 복원하는지 측정.

---

## 3. Reparameterization Trick

### 3.1 문제: 미분 불가
- ELBO의 Reconstruction Term을 계산하려면 sampling된 $z$를 통해 **Backpropagation** 시켜야 함.
- 하지만, $z \sim q_{\phi}(z|x)$는 random process이므로 파라미터 $\phi$에 대해 미분할 수 없음 (Gradient 전파 불가).

### 3.2 해결: Reparameterization
- $z$를 직접 sampling 하는 것이 아닌, auxiliary noise variable $\epsilon$을 사용해서 $z$를 deterministic function으로 재정의함.
    - **기존**: $z \sim N(\mu, \sigma^2 I)$
    - **변경**: $z = \mu + \sigma \odot \epsilon \quad (\text{where } \epsilon \sim N(0, I))$
- Sampling을 $\phi$와 분리하였기에, $z$는 $\mu$와 $\sigma$에 대해 미분 가능한 함수가 되었음 (Backpropagation 가능).

---

## 4. SGVB, AEVB

### 4.1 SGVB (Stochastic Gradient Variational Bayes) Estimator
Reparameterization Trick과 Monte Carlo sampling을 적용하여 Reconstruction Term을 재정의함.
$$\mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(x|z)] \approx \frac{1}{L} \sum_{l=1}^{L} \log p_{\theta}(x|z^{(l)})$$

### 4.2 AEVB (Auto-Encoding Variational Bayes) Algorithm
- Minibatch $M$이 충분히 크다면, SGVB를 통해 얻은 식에서 sampling을 한 번만 진행해도 괜찮음 ($L=1$).
- 전체 Loss는 다음과 같이 근사됨:
  $$\mathcal{L} \approx \frac{N}{M} \sum_{i=1}^{M} \tilde{\mathcal{L}}(x^{(i)})$$
  여기서 개별 데이터의 Loss $\tilde{\mathcal{L}}$은:
  $$\tilde{\mathcal{L}} = \frac{1}{2} \sum (1 + \log(\sigma^2) - \mu^2 - \sigma^2) + \log p_{\theta}(x|z)$$
- $p_{\theta}(x|z)$는 데이터 분포에 따라 Bernoulli (Cross Entropy) 혹은 Gaussian (Mean Squared Error) 등으로 수식화 가능.
- 위 식을 활용하여 parameters를 Backpropagation을 통해 training 시킬 수 있음 (모든 수식이 $\theta, \phi$에 관해 미분 가능).

---

## 5. VAE Architecture

### 5.1 Encoder $q_{\phi}(z|x)$
- **Input**: $x$
- **구조**: MLP (Hidden layer $\rightarrow$ Output layer)
- **Output**: $z$의 distribution parameters인 $\mu$와 $\log(\sigma^2)$

### 5.2 Decoder $p_{\theta}(x|z)$
- **Input**: $z = \mu + \sigma \odot \epsilon$ (where $\epsilon \sim N(0, I)$)
- **구조**: MLP
- **Output**: $x$의 distribution parameters
    - Binary Data: $y$ (Sigmoid activation) $\rightarrow$ **Cross Entropy Loss** 활용
    - Real-valued Data: $\mu, \sigma^2$ $\rightarrow$ **Mean Squared Error** 활용

---

## 6. Conclusion

**VAE의 의의**
1.  **단일 목적 함수**: 기존 알고리즘(Wake-Sleep)과 달리, 단일한 Lower bound를 최적화하여 Encoder, Decoder 동시 학습 가능.
2.  **효율적 역전파**: Reparameterization Trick을 통해 표준적인 SGD (or Adagrad)로 학습 가능.
3.  **자연스러운 정규화**: Auto-Encoder 구조이면서, KL Divergence 항이 포함되어 있어 별도의 regularization term 필요 없이 Overfitting을 방지하고 Latent space 학습 가능.
