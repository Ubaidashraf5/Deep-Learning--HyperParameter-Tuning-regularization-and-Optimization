# 🧠 Deep Learning: Hyperparameter Tuning, Regularization & Optimization

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![DeepLearning.AI](https://img.shields.io/badge/DeepLearning.AI-Specialization-00a3e0.svg)](https://www.deeplearning.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Practical implementations, mathematical foundations, and algorithmic optimizations for training robust, high-performance deep neural networks. Part of the **Deep Learning Specialization** by **deeplearning.ai** and **Andrew Ng**.

---

## 📌 Repository Overview

Building a deep neural network is only the first step. Making it converge quickly, generalize well to unseen data, and train reliably without exploding or vanishing gradients requires mastering the art and science of **hyperparameter tuning, regularization, and modern optimization algorithms**.

This repository organizes the programming assignments into self-contained, concept-driven modules. Each directory includes the notebook, helper modules, visualization assets, and datasets required to understand, reproduce, and experiment with these core techniques.

---

## 🗺️ Curriculum & Topic Breakdown

```
├── Week 01 - Practical Aspects of Deep Learning
│   ├── 01 - Weight Initialization      → Zero, Random, and He / Xavier Initialization
│   ├── 02 - Regularization             → L2 Weight Decay & Inverted Dropout
│   └── 03 - Gradient Checking          → Numerical Gradient Verification via Finite Differences
│
├── Week 02 - Optimization Algorithms
│   └── 01 - Optimization Methods       → Mini-Batch GD, Momentum, RMSprop, Adam & LR Decay
│
└── Week 03 - Introduction to TensorFlow
    └── 01 - TensorFlow Basics and Deep NN → TF 2.x, GradientTape, tf.data & Sign Classifier
```

---

## 🔬 In-Depth Concepts & Module Guide

### Week 01: Practical Aspects of Deep Learning

#### 1. [Weight Initialization](Week%2001%20-%20Practical%20Aspects%20of%20Deep%20Learning/01%20-%20Weight%20Initialization/Weight_Initialization.ipynb)
*Notebook:* `Weight_Initialization.ipynb`

- **The Problem:** Poor weight initialization causes the activation outputs to either vanish (diminish to zero) or explode (overflow towards infinity) across deep layers, preventing gradients from flowing back during backpropagation.
- **Implementations:**
  - **Zero Initialization ($W = 0$):** Fails because all hidden units compute identical outputs ($\text{symmetry problem}$), resulting in identical gradients and preventing the network from learning diverse features.
  - **Large Random Initialization ($W \sim \mathcal{N}(0, 1) \times 10$):** Leads to exploding activations and saturated non-linearities (e.g., sigmoid/tanh saturation), resulting in vanishing gradients and slow learning.
  - **He Initialization ($W^{[l]} \sim \mathcal{N}(0, 1) \times \sqrt{\frac{2}{n^{[l-1]}}}$):** Specifically derived for ReLU activations to maintain variance of activations roughly constant across layers, enabling deep networks to converge reliably.

---

#### 2. [Regularization: L2 & Dropout](Week%2001%20-%20Practical%20Aspects%20of%20Deep%20Learning/02%20-%20Regularization/Regularization_L2_and_Dropout.ipynb)
*Notebook:* `Regularization_L2_and_Dropout.ipynb`

- **The Problem:** Deep networks have immense capacity and tend to overfit small training sets by memorizing noise rather than learning generalizable decision boundaries.
- **Implementations:**
  - **L2 Regularization (Weight Decay):** Augments the cost function with the Frobenius norm penalty:
    $$\mathcal{J}_{\text{regularized}} = \mathcal{J} + \frac{\lambda}{2m} \sum_{l=1}^{L} \|W^{[l]}\|_F^2$$
    This penalizes large weights during backpropagation ($dW^{[l]} = dW_{\text{standard}}^{[l]} + \frac{\lambda}{m} W^{[l]}$), shrinking parameter values towards zero and producing smoother, less complex decision surfaces.
  - **Inverted Dropout:** During training, randomly deactivates each neuron with probability $1 - \text{keep\_prob}$ and scales the remaining activations by $\frac{1}{\text{keep\_prob}}$:
    $$A^{[l]} = \frac{A^{[l]} * D^{[l]}}{\text{keep\_prob}}$$
    This prevents neurons from relying on specific co-adaptations, forcing the network to learn robust, distributed representations. Inverted dropout ensures test-time computation requires no additional scaling.

---

#### 3. [Gradient Checking](Week%2001%20-%20Practical%20Aspects%20of%20Deep%20Learning/03%20-%20Gradient%20Checking/Gradient_Checking.ipynb)
*Notebook:* `Gradient_Checking.ipynb`

- **The Problem:** Analytical backpropagation implementations can easily contain subtle mathematical or indexing errors that still allow the loss to decrease, but prevent the model from reaching optimal performance.
- **Implementation:**
  - Computes numerical gradient approximations using centered finite differences:
    $$\frac{\partial \mathcal{J}}{\partial \theta_i} \approx \frac{\mathcal{J}(\theta_1, \dots, \theta_i + \epsilon, \dots) - \mathcal{J}(\theta_1, \dots, \theta_i - \epsilon, \dots)}{2\epsilon}$$
  - Computes the normalized Euclidean relative difference:
    $$\text{difference} = \frac{\| \text{grad} - \text{gradapprox} \|_2}{\| \text{grad} \|_2 + \| \text{gradapprox} \|_2}$$
  - Verifies that $\text{difference} < 10^{-7}$, confirming the exactness of the analytical gradient implementation.

---

### Week 02: Optimization Algorithms

#### 1. [Optimization Methods: Momentum, RMSprop & Adam](Week%2002%20-%20Optimization%20Algorithms/01%20-%20Optimization%20Methods/Optimization_Methods_Momentum_and_Adam.ipynb)
*Notebook:* `Optimization_Methods_Momentum_and_Adam.ipynb`

- **The Problem:** Batch Gradient Descent is slow on massive datasets and prone to pathological oscillations along steep ravines in parameter space.
- **Implementations:**
  - **Mini-Batch Gradient Descent:** Splits data into minibatches (e.g., 64, 128, 256) after random shuffling, allowing frequent parameter updates and taking advantage of vectorization.
  - **Momentum:** Computes an exponentially weighted moving average of the gradients:
    $$v_{dW} = \beta v_{dW} + (1 - \beta) dW, \quad W := W - \alpha v_{dW}$$
    Dampens oscillations in high-curvature directions and accelerates along low-curvature directions.
  - **RMSprop (Root Mean Square Propagation):** Normalizes updates by the running average of squared gradients:
    $$s_{dW} = \beta_2 s_{dW} + (1 - \beta_2) (dW)^2, \quad W := W - \alpha \frac{dW}{\sqrt{s_{dW}} + \varepsilon}$$
  - **Adam (Adaptive Moment Estimation):** Combines Momentum (first moment $v$) and RMSprop (second moment $s$) with bias correction for initial steps:
    $$v_{dW}^{\text{corrected}} = \frac{v_{dW}}{1 - \beta_1^t}, \quad s_{dW}^{\text{corrected}} = \frac{s_{dW}}{1 - \beta_2^t}$$
    $$W := W - \alpha \frac{v_{dW}^{\text{corrected}}}{\sqrt{s_{dW}^{\text{corrected}}} + \varepsilon}$$
  - **Learning Rate Decay:** Implements scheduled learning rate decay ($\alpha = \frac{\alpha_0}{1 + \text{decay\_rate} \times \text{epoch}}$) to stabilize parameters around the global minimum.

---

### Week 03: Introduction to TensorFlow 2.x

#### 1. [TensorFlow 2.x & Deep Neural Networks](Week%2003%20-%20Introduction%20to%20TensorFlow/01%20-%20TensorFlow%20Basics%20and%20Deep%20NN/Introduction_to_TensorFlow2_and_Deep_NN.ipynb)
*Notebook:* `Introduction_to_TensorFlow2_and_Deep_NN.ipynb`

- **The Transition:** Moving from manual NumPy matrix derivations to an industry-grade automatic differentiation framework.
- **Key Concepts Covered:**
  - Eager execution and tensor manipulation with `tf.constant`, `tf.Variable`, and `tf.cast`.
  - Automatic differentiation via `tf.GradientTape()` context manager.
  - Efficient input pipelining with `tf.data.Dataset` (`.map()`, `.batch()`, `.prefetch()`).
  - One-hot encoding using `tf.one_hot()`.
  - Building, compiling, and training a 3-layer Deep Neural Network from scratch in TensorFlow 2 for multi-class classification on the **SIGNS dataset** (hand signs representing numbers 0 through 5).

---

## 📂 Repository Structure

```plaintext
.
├── .gitattributes
├── .gitignore
├── README.md
├── requirements.txt
│
├── Week 01 - Practical Aspects of Deep Learning/
│   ├── 01 - Weight Initialization/
│   │   ├── Weight_Initialization.ipynb
│   │   ├── init_utils.py
│   │   └── public_tests.py
│   │
│   ├── 02 - Regularization/
│   │   ├── Regularization_L2_and_Dropout.ipynb
│   │   ├── reg_utils.py
│   │   ├── testCases.py
│   │   ├── public_tests.py
│   │   ├── requirements.txt
│   │   ├── datasets/
│   │   └── images/
│   │
│   └── 03 - Gradient Checking/
│       ├── Gradient_Checking.ipynb
│       ├── gc_utils.py
│       ├── testCases.py
│       ├── public_tests.py
│       └── images/
│
├── Week 02 - Optimization Algorithms/
│   └── 01 - Optimization Methods/
│       ├── Optimization_Methods_Momentum_and_Adam.ipynb
│       ├── opt_utils_v1a.py
│       ├── testCases.py
│       ├── public_tests.py
│       ├── datasets/
│       └── images/
│
└── Week 03 - Introduction to TensorFlow/
    └── 01 - TensorFlow Basics and Deep NN/
        ├── Introduction_to_TensorFlow2_and_Deep_NN.ipynb
        ├── tf_utils.py
        ├── improv_utils.py
        ├── test_utils.py
        ├── public_tests.py
        ├── datasets/
        └── images/
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.8+ (Python 3.9 or 3.10 recommended)
- `pip` or `conda`

### 2. Clone the Repository
```bash
git clone https://github.com/Ubaidashraf5/Deep-Learning--HyperParameter-Tuning-regularization-and-Optimization.git
cd Deep-Learning--HyperParameter-Tuning-regularization-and-Optimization
```

### 3. Create a Virtual Environment
```bash
# Using venv (macOS/Linux)
python3 -m venv venv
source venv/bin/activate

# Using venv (Windows PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Launch Jupyter Lab / Notebook
```bash
jupyter notebook
```
Navigate to any week's directory and run the notebooks interactively.

---

## 🛠️ Tech Stack & Key Libraries

| Technology | Purpose |
|---|---|
| **Python 3** | Core language for scientific computing and modeling |
| **NumPy** | High-performance N-dimensional array manipulations |
| **TensorFlow 2.x** | Automatic differentiation, computational graphs, and neural networks |
| **Matplotlib** | Loss curve visualization and 2D decision boundary rendering |
| **SciPy & Scikit-Learn** | Scientific utilities, dataset generation, and matrix loading |
| **H5py** | Reading `.h5` binary format image datasets |
| **Jupyter** | Interactive development and documentation |

---

## 📚 Acknowledgments & References

- **Course:** [Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization](https://www.coursera.org/learn/deep-neural-network) by **DeepLearning.AI**
- **Instructor:** **Andrew Ng**
- Special thanks to the deeplearning.ai teaching team for designing comprehensive labs and datasets.

---

## 📄 License
This repository is shared for educational and personal study purposes.
