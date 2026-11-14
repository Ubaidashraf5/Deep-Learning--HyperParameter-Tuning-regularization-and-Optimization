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

---

#### 1. [Weight Initialization](Week%2001%20-%20Practical%20Aspects%20of%20Deep%20Learning/01%20-%20Weight%20Initialization/Weight_Initialization.ipynb)
*Notebook:* `Weight_Initialization.ipynb`

**The Challenge:**
Poor weight initialization causes activations to either vanish (diminish to zero) or explode (overflow towards infinity) across deep layers, preventing gradients from flowing back during backpropagation.

**Techniques Implemented:**

- **Zero Initialization ($W = 0$):**
  Setting weights to zero fails because every hidden unit computes identical activations ($a_1^{[l]} = a_2^{[l]} = \dots$), meaning all neurons compute the exact same feature. Gradients remain identical across neurons, completely preventing the network from breaking symmetry.

- **Large Random Initialization ($W \sim \mathcal{N}(0, 1) \times 10$):**
  Multiplying Gaussian random weights by large constants produces extreme activation values. For sigmoid or tanh units, large inputs push activations into saturation regions ($\approx \pm 1$) where the derivative $\sigma'(z) \approx 0$, causing vanishing gradients and stalling training.

- **He Initialization:**
  For a layer $l$ with $n^{[l-1]}$ input features, weights are drawn from a normal distribution scaled by the number of inputs:

$$
W^{[l]} \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n^{[l-1]}}}\right)
$$

  Specifically derived for ReLU activations ($g(z) = \max(0, z)$), He initialization preserves the variance of activations and backpropagated gradients throughout all layers, enabling deep architectures to converge smoothly.

---

#### 2. [Regularization: L2 & Dropout](Week%2001%20-%20Practical%20Aspects%20of%20Deep%20Learning/02%20-%20Regularization/Regularization_L2_and_Dropout.ipynb)
*Notebook:* `Regularization_L2_and_Dropout.ipynb`

**The Challenge:**
Deep neural networks have high capacity and easily overfit small training datasets by memorizing noise rather than learning generalizable decision boundaries.

**Techniques Implemented:**

- **L2 Regularization (Weight Decay):**
  Augments the standard cross-entropy cost function with a penalty proportional to the Frobenius norm of all weight matrices:

$$
\mathcal{J}_{\text{regularized}} = \mathcal{J} + \frac{\lambda}{2m} \sum_{l=1}^{L} \|W^{[l]}\|_F^2
$$

  During backward propagation, the gradient calculation incorporates this penalty:

$$
dW^{[l]} = dW_{\text{standard}}^{[l]} + \frac{\lambda}{m} W^{[l]}
$$

  This effectively updates weights as $W^{[l]} := W^{[l]}\left(1 - \frac{\alpha \lambda}{m}\right) - \alpha dW_{\text{standard}}^{[l]}$, shrinking weights towards zero on each iteration (weight decay) to prevent individual neurons from exerting disproportionate influence.

- **Inverted Dropout:**
  During training, each hidden neuron is independently retained with probability $p$ (`keep_prob`) and silenced with probability $1 - p$. Forward activations are gated with a random binary dropout mask $D^{[l]} \sim \text{Bernoulli}(p)$:

$$
A^{[l]} = \frac{A^{[l]} \odot D^{[l]}}{p}
$$

  By dividing by $p$ during training, the expected value $\mathbb{E}[A^{[l]}]$ remains identical to non-dropout activations. Consequently, **no modification or scaling is needed at test time**. Inverted dropout forces neurons to learn robust features without relying on fragile co-adaptations.

---

#### 3. [Gradient Checking](Week%2001%20-%20Practical%20Aspects%20of%20Deep%20Learning/03%20-%20Gradient%20Checking/Gradient_Checking.ipynb)
*Notebook:* `Gradient_Checking.ipynb`

**The Challenge:**
Analytical backpropagation code is notoriously prone to subtle indexing, transposition, or sign bugs that may still decrease the cost function but converge to suboptimal solutions.

**Technique Implemented:**

- Computes two-sided finite difference approximations for each parameter $\theta_i$:

$$
\frac{\partial \mathcal{J}}{\partial \theta_i} \approx \frac{\mathcal{J}(\theta_1, \dots, \theta_i + \epsilon, \dots) - \mathcal{J}(\theta_1, \dots, \theta_i - \epsilon, \dots)}{2\epsilon}
$$

- Evaluates the normalized relative Euclidean distance between the numerical approximation vector $d\theta_{\text{approx}}$ and the analytical gradient vector $d\theta$:

$$
\text{difference} = \frac{\| d\theta - d\theta_{\text{approx}} \|_2}{\| d\theta \|_2 + \| d\theta_{\text{approx}} \|_2}
$$

- A calculated value of $\text{difference} < 10^{-7}$ confirms that the analytical gradient computation is mathematically exact.

---

### Week 02: Optimization Algorithms

---

#### 1. [Optimization Methods: Momentum, RMSprop & Adam](Week%2002%20-%20Optimization%20Algorithms/01%20-%20Optimization%20Methods/Optimization_Methods_Momentum_and_Adam.ipynb)
*Notebook:* `Optimization_Methods_Momentum_and_Adam.ipynb`

**The Challenge:**
Standard Batch Gradient Descent computes gradients across the entire dataset before every step, leading to sluggish progress on large datasets and high-variance oscillations along steep ravine surfaces.

**Techniques Implemented:**

- **Mini-Batch Gradient Descent:**
  Splits training data into randomly shuffled mini-batches of size $B$ (e.g., 64, 128, 256), taking parameter steps after each mini-batch to combine vectorized hardware efficiency with rapid progress.

- **Gradient Descent with Momentum:**
  Tracks an exponentially weighted moving average of past gradients to build velocity $v$:

$$
v_{dW} = \beta v_{dW} + (1 - \beta) dW
$$

$$
W := W - \alpha v_{dW}
$$

  Momentum dampens high-frequency vertical oscillations while accelerating progress along the horizontal path towards the minimum.

- **RMSprop (Root Mean Square Propagation):**
  Divides parameter updates by the running square root of past squared gradients $s$:

$$
s_{dW} = \beta_2 s_{dW} + (1 - \beta_2) (dW)^2
$$

$$
W := W - \alpha \frac{dW}{\sqrt{s_{dW}} + \varepsilon}
$$

- **Adam (Adaptive Moment Estimation):**
  Combines the principles of Momentum (first moment $v$) and RMSprop (second moment $s$), incorporating bias correction for early iterations:

$$
v_{dW}^{\text{corrected}} = \frac{v_{dW}}{1 - \beta_1^t}, \quad s_{dW}^{\text{corrected}} = \frac{s_{dW}}{1 - \beta_2^t}
$$

$$
W := W - \alpha \frac{v_{dW}^{\text{corrected}}}{\sqrt{s_{dW}^{\text{corrected}}} + \varepsilon}
$$

- **Learning Rate Decay:**
  Schedules dynamic learning rate reduction over training epochs to prevent parameters from overshooting around the global optimum:

$$
\alpha = \frac{\alpha_0}{1 + k \times \text{epoch}}
$$

  where $k$ represents the decay rate hyperparameter (`decay_rate`).

---

### Week 03: Introduction to TensorFlow 2.x

---

#### 1. [TensorFlow 2.x & Deep Neural Networks](Week%2003%20-%20Introduction%20to%20TensorFlow/01%20-%20TensorFlow%20Basics%20and%20Deep%20NN/Introduction_to_TensorFlow2_and_Deep_NN.ipynb)
*Notebook:* `Introduction_to_TensorFlow2_and_Deep_NN.ipynb`

**The Transition:**
Moving from low-level NumPy matrix derivations to an industry-grade automatic differentiation framework.

**Key Concepts Covered:**

- **Tensors & Variables:** Eager execution fundamentals with `tf.constant`, mutable `tf.Variable`, and type casting via `tf.cast`.
- **Automatic Differentiation:** Recording forward operations with the `tf.GradientTape()` context manager to automatically compute analytical gradients:

$$
\nabla_{\theta} \mathcal{J}(\theta)
$$

- **Data Streaming Pipeline:** Utilizing `tf.data.Dataset` transformations (`.map()`, `.batch()`, `.prefetch()`) to eliminate I/O bottlenecks by pre-loading mini-batches into memory during GPU/CPU execution.
- **End-to-End Deep Neural Network:** Building, compiling, and training a 3-layer neural network with Adam optimization and Categorical Crossentropy to classify sign language numbers (0 through 5) from the **SIGNS dataset**.

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
