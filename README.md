# Speech-Based Parkinson’s Disease Analysis
### From Acoustic Feature Engineering to Physics-Inspired Vision Transformers

<p align="center">
  <strong>A unified speech-processing and deep-learning framework for Parkinson’s disease analysis</strong>
</p>

<p align="center">
  <em>Acoustic Features • Self-Supervised Learning • Spectrograms • Vision Transformers • Physics-Inspired Dynamics • Domain-Adversarial Learning</em>
</p>

---

## 📌 Overview

Speech is a rich biological signal containing information about **phonation, articulation, vocal-tract configuration, timing, and motor control**.

This project develops a progression from conventional speech processing to a physics-inspired deep-learning framework for **Parkinson’s disease (PD) analysis**.

The work combines:

- **Classical acoustic feature engineering**
- **Frequency-domain and cepstral analysis**
- **Classical machine learning**
- **Self-supervised speech representation learning**
- **Learned speech embeddings**
- **LSTM and Transformer-based sequence modeling**
- **Spectrogram-based Vision Transformers**
- **Physics-inspired dynamical modeling**
- **Gradient Reversal Layer (GRL) based domain-adversarial learning**

The overall research progression is:

```text
Speech
   ↓
Acoustic / Spectral Representation
   ↓
Learned Representation
   ↓
Structured Dynamics
   ↓
Robust Classification
```

---

## 🎯 Research Motivation

Parkinson’s disease is associated with motor impairments that can affect speech production.

Potential speech manifestations include:

- Reduced loudness
- Monopitch
- Reduced intensity variation
- Imprecise articulation
- Altered timing
- Pauses
- Vocal instability

These changes motivate speech as a **non-invasive signal for computational PD analysis**.

The project therefore asks:

> **Can interpretable acoustic measurements and learned time-frequency representations be combined with structured dynamical modeling to obtain robust representations for PD analysis?**

---

# 🧭 Research Pipeline

The framework progresses through two complementary perspectives.

### Conventional Speech-Processing Pipeline

```text
Speech
  ↓
Preprocessing
  ↓
Segmentation
  ↓
Feature Extraction
  ↓
ML / DL
  ↓
Prediction
```

### Deep-Learning Pipeline

```text
Speech
  ↓
Spectrogram
  ↓
Encoder
  ↓
Representation
  ↓
Prediction
```

The two approaches are complementary:

**Hand-crafted features** provide interpretable acoustic descriptors, while **learned representations** can capture complex patterns that are difficult to specify manually.

---

# 🗂️ Project Structure

The research can be viewed as the following progression:

```text
                    SPEECH SIGNAL
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
   SIGNAL PROCESSING              DEEP LEARNING
          │                             │
          ▼                             ▼
 Classical Features               Spectrogram
          │                             │
          ▼                             ▼
    ML Classifiers                ConvStem + ViT
          │                             │
          │                             ▼
          │                         h_CLS
          │                             │
          └──────────────┬──────────────┘
                         ▼
               Structured Dynamics
                         │
                         ▼
              Domain-Invariant Features
                         │
                         ▼
                   PD Analysis
```

---

# 1. 🎙️ Speech Signal Representation

An analog speech signal is represented as:

\[
x(t)
\]

After sampling:

\[
x[n] = x(nT_s),
\qquad
f_s = \frac{1}{T_s}.
\]

For a recording of duration \(T\), the approximate number of samples is:

\[
N=f_sT.
\]

The Nyquist condition requires:

\[
f_{\max}<\frac{f_s}{2}.
\]

For example, a 16-kHz recording can represent frequencies up to approximately **8 kHz**.

Bit depth controls amplitude quantization. With \(B\) bits, there are:

\[
2^B
\]

possible quantization levels.

---

# 2. 🧹 Preprocessing

Raw recordings may contain:

- Silence
- Environmental noise
- Microphone noise
- Electrical interference
- Reverberation
- Speech
- Non-speech events

A practical preprocessing pipeline is:

```text
Raw Recording
     ↓
Silence Removal
     ↓
Noise Reduction
     ↓
Normalization
```

Normalization is expressed as:

\[
x_{\mathrm{norm}}
=
\frac{x-\mu}{\sigma}.
\]

The normalization statistics should be computed using the appropriate training data to avoid information leakage.

---

# 3. 🎤 Voice Activity Detection, Framing & Windowing

## Voice Activity Detection

Voice Activity Detection (VAD) identifies speech-containing regions.

Short-time RMS energy can be defined as:

\[
E_{\mathrm{RMS}}[m]
=
\sqrt{
\frac{1}{N}
\sum_{n=0}^{N-1}x_m[n]^2
}.
\]

A frame may be classified as non-speech when:

\[
E_{\mathrm{RMS}}[m]<\tau.
\]

Zero-crossing rate is:

\[
ZCR
=
\frac{1}{N-1}
\sum_{n=1}^{N-1}
\mathbf{1}[x[n]x[n-1]<0].
\]

Energy and ZCR can be combined in practical VAD systems.

## Framing

Speech is approximately stationary only over short intervals, so it is divided into overlapping frames:

\[
x_m[n]=x[n+mH],
\]

where \(H\) is the hop size.

Windowing gives:

\[
x_w[n]=x_m[n]w[n].
\]

For a Hamming window:

\[
w[n]
=
0.54-0.46
\cos\left(
\frac{2\pi n}{N-1}
\right).
\]

Windowing reduces discontinuities at frame boundaries and spectral leakage.

---

# 4. 🎵 Traditional Acoustic Features

A major part of the framework uses interpretable speech descriptors.

## Fundamental Frequency \(F_0\)

The fundamental frequency describes the rate of vocal-fold vibration:

\[
F_0=\frac{1}{T_0}.
\]

Pitch is the perceptual correlate of \(F_0\).

Mean pitch:

\[
\bar F_0
=
\frac{1}{M}
\sum_{m=1}^{M}F_0^{(m)}.
\]

Pitch variability:

\[
\sigma_{F_0}
=
\sqrt{
\frac{1}{M}
\sum_{m=1}^{M}
\left(
F_0^{(m)}-\bar F_0
\right)^2
}.
\]

Reduced pitch variability may be associated with monopitch in Parkinsonian speech.

---

## Jitter

Jitter describes cycle-to-cycle variation in vocal period:

\[
J_{\mathrm{abs}}
=
\frac{1}{N-1}
\sum_{i=1}^{N-1}
|T_i-T_{i+1}|.
\]

**Jitter → frequency / period variation**

---

## Shimmer

Shimmer describes amplitude variation:

\[
S_{\mathrm{abs}}
=
\frac{1}{N-1}
\sum_{i=1}^{N-1}
|A_i-A_{i+1}|.
\]

**Shimmer → amplitude variation**

---

## HNR

The harmonics-to-noise ratio is conceptually:

\[
HNR_{\mathrm{dB}}
=
10\log_{10}
\left(
\frac{P_{\mathrm{harmonic}}}
{P_{\mathrm{noise}}}
\right).
\]

---

## Formants

Formants are vocal-tract resonances:

\[
F_1,F_2,F_3,\ldots
\]

They are affected by the configuration of the tongue, lips, jaw, and other vocal-tract structures.

---

## RMS Energy

\[
RMS
=
\sqrt{
\frac{1}{N}
\sum_{n=1}^{N}x[n]^2
}.
\]

Useful energy descriptors include:

- Mean RMS
- RMS standard deviation
- Extrema
- Dynamic range
- Temporal energy patterns

---

# 5. 🌐 Frequency-Domain Analysis

The Fourier transform expresses a signal in terms of frequency components:

\[
X(f)
=
\int_{-\infty}^{\infty}
x(t)e^{-j2\pi ft}\,dt.
\]

For a discrete signal:

\[
X[k]
=
\sum_{n=0}^{N-1}
x[n]e^{-j2\pi kn/N}.
\]

A direct DFT has approximately:

\[
O(N^2)
\]

complexity, whereas FFT algorithms reduce this to approximately:

\[
O(N\log N).
\]

---

# 6. 🎼 DCT, Cepstral Analysis & MFCCs

The Discrete Cosine Transform (DCT) represents a sequence using cosine basis functions.

It is particularly important for MFCCs because it approximately decorrelates filter-bank energies and concentrates information into a compact representation.

## MFCC Pipeline

```text
Waveform
   ↓
Framing
   ↓
Windowing
   ↓
FFT
   ↓
Power Spectrum
   ↓
Mel Filter Bank
   ↓
Log Energies
   ↓
DCT
   ↓
MFCCs
```

A common mel-scale conversion is:

\[
m(f)
=
2595
\log_{10}
\left(
1+\frac{f}{700}
\right).
\]

For a filter \(H_m[k]\) and power spectrum \(P[k]\):

\[
E_m
=
\sum_kP[k]H_m[k].
\]

Log filter-bank energy:

\[
L_m=\log(E_m+\epsilon).
\]

The DCT produces MFCC coefficients:

\[
c_n
=
\sum_{m=1}^{M}
L_m
\cos
\left[
\frac{\pi n}{M}
\left(
m-\frac12
\right)
\right].
\]

A representative classical feature vector is:

\[
\begin{aligned}
[
&F_0,\,
\sigma_{F_0},\,
\mathrm{Jitter},\,
\mathrm{Shimmer},\,
\mathrm{HNR},\,
\mathrm{RMS},\,
\mathrm{ZCR},\\
&F_1,\,
F_2,\,
F_3,\,
\mathrm{MFCC}_1,\,
\ldots,\,
\mathrm{MFCC}_K
].
\end{aligned}
\]

---

# 7. 🤖 Classical & Learned Speech Representations

## Classical Machine Learning

Classical models operate on engineered acoustic features.

### Random Forest

Random Forest uses an ensemble of decision trees:

\[
\hat y
=
\operatorname{mode}
\{T_1(x),\ldots,T_B(x)\}.
\]

### SVM

An SVM seeks a decision boundary with a large margin.

### Feed-Forward Neural Network

A neural network can also classify the acoustic feature vector directly.

---

## Self-Supervised Learning

Self-supervised learning (SSL) learns representations without requiring a manual label for every training example.

The conceptual pipeline is:

```text
Unlabeled Speech
      ↓
SSL Pretraining
      ↓
Speech Embedding
      ↓
PD Classifier
```

A wav2vec-style system can be summarized as:

```text
Waveform
   ↓
Feature Encoder
   ↓
Context Network
   ↓
Contextual Representation
```

A learned embedding such as:

\[
e\in\mathbb{R}^{768}
\]

should **not** be interpreted as 768 independent hand-crafted acoustic features. Its dimensions jointly form a learned representation.

Hand-crafted and learned features can also be combined:

\[
h=[e;a],
\qquad
\hat y=f_\theta(h).
\]

---

# 8. 🔄 Sequence Modeling

Speech is inherently sequential.

**LSTMs** model temporal dependencies using recurrent gates, whereas **Transformers** use attention.

Scaled dot-product attention is:

\[
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}
\right)V.
\]

Multi-head attention is:

\[
\operatorname{MHA}(Q,K,V)
=
\operatorname{Concat}
(\mathrm{head}_1,\ldots,\mathrm{head}_h)W^O.
\]

This permits relationships between distant speech frames to be modeled directly.

---

# 9. 🔬 Physics-Inspired Spectrogram Representation

The deep-learning perspective begins with the spectrogram:

\[
X(f,t),
\]

which describes acoustic energy over time and frequency.

If there are 50 frequency bins, a time frame can be represented as:

\[
X_t=
[
X(f_1,t),\ldots,X(f_{50},t)
].
\]

### Important Interpretation

These 50 values are **acoustic frequency bins**, not 50 physical vocal-tract regions.

The conceptual relationship is:

```text
Articulator Movement
        ↓
Vocal-Tract Configuration
        ↓
Acoustic Response
        ↓
Frequency Distribution
```

Different parts of the vocal tract have different acoustic effects, while the overall vocal-tract configuration determines how acoustic energy is distributed across frequency regions.

---

# 10. ⚙️ Physics-Inspired Dynamical Model

The acoustic dynamics are represented by the abstraction:

\[
\frac{d^2X_f}{dt^2}
+
2\gamma_f
\frac{dX_f}{dt}
+
\omega_f^2X_f
=
F.
\]

Here:

| Quantity | Interpretation |
|---|---|
| \(F\) | Driving force / excitation |
| \(\gamma\) | Damping-related behavior |
| \(\omega\) | Oscillatory / natural-frequency-related behavior |
| \(X_f\) | Response associated with frequency component \(f\) |

The equation is a **learned dynamical abstraction**, rather than a claim that every spectrogram bin is an independent physical oscillator.

---

## Why is \(F\) Shared?

The architectural assumption is:

\[
F(t)\rightarrow\text{shared excitation}.
\]

Meanwhile:

\[
\gamma(t)
=
[
\gamma_1(t),\ldots,\gamma_{50}(t)
]
\]

and

\[
\omega(t)
=
[
\omega_1(t),\ldots,\omega_{50}(t)
]
\]

are frequency dependent.

Therefore:

```text
                 Shared Excitation
                       F(t)
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
         f₁             f₂            f₅₀
          │              │              │
     γ₁(t), ω₁(t)  γ₂(t), ω₂(t)  γ₅₀(t), ω₅₀(t)
```

The 50 outputs correspond to **acoustic frequency bins**, not anatomical regions.

---

# 11. 📈 Dynamical Regularization

The learned force can become unnecessarily irregular.

A smoothness penalty is:

\[
L_{\mathrm{smooth}}^F
=
\sum_t
|F_{t+1}-F_t|^2.
\]

A gradient-difference penalty is:

\[
L_{\mathrm{grad}}
=
\sum_t
\left[
(F_{t+1}-F_t)
-
(F_t-F_{t-1})
\right]^2.
\]

Equivalently:

\[
L_{\mathrm{grad}}
=
\sum_t
(F_{t+1}-2F_t+F_{t-1})^2.
\]

### Interpretation

- **Smoothness loss:** controls first-order temporal variation.
- **Gradient-difference loss:** controls changes in the temporal slope.

Together, they encourage smoother learned dynamics.

---

# 12. 👁️ Convolutional Stem + Vision Transformer

The complete feature-extraction path is:

```text
Speech
  ↓
Spectrogram
  ↓
ConvStem
  ↓
GELU
  ↓
Stride Downsampling
  ↓
Patch / Token Representation
  ↓
Vision Transformer
  ↓
h_CLS
```

## Convolutional Stem

The convolutional stem extracts **local time-frequency patterns** and reduces resolution.

Stride-based downsampling reduces the number of tokens and therefore the computational cost of self-attention.

GELU is:

\[
GELU(x)=x\Phi(x),
\]

where \(\Phi(x)\) is the standard Gaussian CDF.

---

## Patch Representation

A spectrogram can be divided into patches:

\[
[P_1,P_2,\ldots,P_N].
\]

A learnable CLS token is prepended:

\[
[\mathrm{CLS},P_1,\ldots,P_N].
\]

After Transformer processing:

\[
[\mathrm{CLS},P_1,\ldots,P_N]
\rightarrow
[h_{\mathrm{CLS}},h_1,\ldots,h_N].
\]

The CLS representation is:

\[
h_{\mathrm{CLS}}.
\]

### What is \(h_{\mathrm{CLS}}\)?

\(h_{\mathrm{CLS}}\) acts as a **global representation of the input spectrogram**.

It aggregates information from different time-frequency regions through self-attention.

For binary classification:

\[
h_{\mathrm{CLS}}
\rightarrow
\text{Linear Layer}
\rightarrow
[P(\mathrm{HC}),P(\mathrm{PD})].
\]

The **HC/PD prediction is produced by the classification head**, not by \(h_{\mathrm{CLS}}\) itself.

---

# 13. 🧭 Positional Encoding vs Causality

These are two different concepts.

```text
Positional Encoding
        │
        └──► tells the Transformer WHERE a token occurs

Causal Mask
        │
        └──► controls WHICH tokens a token can attend to
```

Causality is imposed by the **attention pattern**, typically using a mask that prevents attention to future positions.

Positional encoding itself is **not causal**.

A standard sinusoidal positional encoding is:

\[
PE(pos,2i)
=
\sin
\left(
\frac{pos}{10000^{2i/d_{\mathrm{model}}}}
\right),
\]

\[
PE(pos,2i+1)
=
\cos
\left(
\frac{pos}{10000^{2i/d_{\mathrm{model}}}}
\right).
\]

---

# 14. 🛡️ Domain-Adversarial Learning

A domain can represent systematic acquisition differences such as:

- Dataset
- Microphone
- Recording environment
- Recording session

The desired representation should rely on speech characteristics relevant to the task rather than nuisance domain characteristics.

The representation \(z\) is sent to two branches:

```text
                    Encoder
                       │
                       ▼
                       z
                    ╱   ╲
                   ╱     ╲
                  ▼       ▼
           HC / PD       GRL
          Classifier      │
                          ▼
                   Domain Classifier
```

---

## Gradient Reversal Layer

During the forward pass:

\[
GRL(z)=z.
\]

During backpropagation:

\[
\frac{\partial GRL}{\partial z}
=
-\lambda I.
\]

Thus, if the domain classifier receives gradient \(g\), the encoder receives:

\[
-\lambda g.
\]

The domain classifier attempts to identify the domain, while the feature extractor is encouraged to make domain identification difficult.

A conceptual objective is:

\[
\min_{\theta_f,\theta_y}
\max_{\theta_d}
\left[
L_{\mathrm{class}}
-
\lambda L_{\mathrm{domain}}
\right].
\]

### Important

**GRL does not classify HC versus PD.**

**GRL does not assign relevance probabilities to individual features.**

Its purpose is to encourage the encoder to learn **domain-invariant representations**.

---

# 15. 🧱 Stochastic Depth & Learning-Rate Scheduling

## Stochastic Depth

Stochastic depth randomly skips complete residual blocks during training.

Unlike dropout, which masks individual activations, stochastic depth removes an entire residual path for a training iteration.

This provides regularization for deep architectures.

---

## Cosine Learning-Rate Schedule

The cosine learning-rate schedule is:

\[
LR_t
=
LR_{\min}
+
\frac12
(LR_{\max}-LR_{\min})
\left[
1+
\cos
\left(
\frac{\pi t}{T}
\right)
\right].
\]

It allows larger updates early in training and smaller refinements later.

The scheduler controls the **learning rate**, rather than directly changing the definition of the loss.

---

# 16. 🔁 Reconstruction, Prediction & Causality

Reconstruction can be used as a representation-learning objective:

```text
Speech
  ↓
Encoder
  ↓
Latent Representation
  ↓
Reconstruction
```

This is particularly useful when labeled data are limited.

The learned representation can subsequently support supervised classification.

---

## Offline vs Real-Time Modeling

Causality must be distinguished from positional encoding.

A causal model uses only permitted past information, whereas a non-causal model may use future context:

\[
x_t
\leftarrow
x_{t-k},\ldots,x_t,\ldots,x_{t+k}.
\]

Therefore:

- **Non-causal / offline:** future context can be used.
- **Causal / real-time:** future information beyond the allowed latency cannot be used.

Claims of causality or real-time operation must therefore be tied to an explicit operational definition.

---

# 17. 🧮 Multi-Objective Training

A conceptual total objective is:

\[
L_{\mathrm{total}}
=
L_{\mathrm{task}}
+
\lambda_{\mathrm{rec}}L_{\mathrm{rec}}
+
\lambda_{\mathrm{smooth}}L_{\mathrm{smooth}}
+
\lambda_{\mathrm{grad}}L_{\mathrm{grad}}
+
\lambda_{\mathrm{adv}}L_{\mathrm{domain}}.
\]

The domain component is implemented adversarially through GRL rather than simply minimized by the feature extractor.

For ordinary binary classification, binary cross-entropy is:

\[
L_{\mathrm{BCE}}
=
-
\left[
y\log p
+
(1-y)\log(1-p)
\right].
\]

The exact coefficients depend on the final implementation.

---

# 18. 🧪 Experimental Design

The framework considers a progression of representations and classifiers:

| Experiment | Representation | Classifier |
|---:|---|---|
| 1 | MFCC only | SVM |
| 2 | Classical acoustic features | Random Forest |
| 3 | Classical acoustic features | MLP |
| 4 | Learned speech embedding | MLP |
| 5 | Learned speech embedding | SVM |
| 6 | Classical + learned | MLP |
| 7 | Frame sequence | LSTM |
| 8 | Frame sequence | Transformer |

This comparison addresses whether learned representations provide information beyond carefully designed acoustic features.

---

# 19. 📊 Evaluation

Evaluation should not rely on accuracy alone.

Relevant metrics include:

- **Precision**
- **Recall / Sensitivity**
- **Specificity**
- **F1 score**
- **ROC curves**
- **AUC**
- Confusion matrix

## Subject-Level Leakage

In PD analysis, **subject-independent splitting is essential**.

Recordings from the same speaker should not unintentionally appear in both training and test sets.

Otherwise, the model may learn speaker-specific characteristics rather than disease-relevant speech characteristics.

---

# 20. 🔗 Integrated End-to-End Framework

The combined research framework can be summarized as:

```text
Speech Waveform
      ↓
Preprocessing + VAD
      ↓
Spectrogram
      ↓
┌─────────────────────────────────┐
│ Physics-Inspired Representation │
│                                 │
│ F-head                          │
│ γ-head                          │
│ ω-head                          │
└─────────────────────────────────┘
      ↓
ConvStem + GELU + Downsampling
      ↓
Patch Embeddings
      ↓
[CLS, P₁, ..., Pₙ]
      ↓
Vision Transformer
      ↓
h_CLS
      │
      ├──────────────► HC / PD Classifier
      │
      └──────────────► GRL → Domain Classifier
```

This architecture combines **four complementary forms of information**:

### 1. Interpretable Acoustic Descriptors

Pitch, jitter, shimmer, HNR, formants, RMS, ZCR, and MFCCs provide recognizable measurements of phonation and articulation.

### 2. Learned Representations

Learned embeddings can encode combinations of information distributed across many dimensions.

### 3. Vision Transformer

The ViT models long-range relationships between time-frequency patches.

### 4. Structured Dynamics + Domain Robustness

The physics-inspired branch models temporal dynamics, while the adversarial branch suppresses domain-specific nuisance information.

---

# 21. 💡 Key Conceptual Takeaways

### Acoustic Features

```text
F₀       → vocal-fold vibration rate
Jitter   → period / frequency variation
Shimmer  → amplitude variation
HNR      → harmonic vs noise structure
F₁,F₂,F₃ → vocal-tract resonances
MFCCs    → compact spectral / cepstral representation
```

### Learned Embeddings

A learned embedding is a **joint representation**, not a list of independent manually defined acoustic features.

### Frequency Bins

```text
50 frequency bins
        ≠
50 anatomical regions
```

They represent different acoustic frequency regions.

### Physics-Inspired Parameters

```text
F(t)             → shared excitation
γ(f,t)           → frequency-dependent damping behavior
ω(f,t)           → frequency-dependent oscillatory behavior
X(f,t)           → acoustic response
```

### CLS Representation

```text
Spectrogram
    ↓
Patch Tokens
    ↓
Vision Transformer
    ↓
h_CLS
    ↓
Classification Head
    ↓
HC / PD Probability
```

### GRL

```text
Encoder
   ↓
Feature Representation
   ├──► Main Task
   │
   └──► GRL ──► Domain Classifier
                   │
                   ▼
             Gradient Reversal
```

GRL encourages **domain-invariant features**; it is not a feature-importance estimator.

---

# 22. ⚠️ Limitations & Practical Considerations

Several sources of error should be considered:

- Recording noise
- Microphone differences
- Room acoustics
- Speaker variability
- Class imbalance
- Data leakage

Aggressive denoising may also remove acoustic characteristics that are useful for diagnosis.

Similarly, a complex model does not automatically provide a stronger experiment than a carefully controlled classical baseline.

The distinction between **offline analysis** and **true real-time prediction** is also important.

A non-causal Transformer can exploit both past and future context, whereas a strict real-time system cannot use future frames beyond its allowed latency.

---

# 23. 🚀 Future Research Directions

The framework naturally motivates several extensions:

- Multimodal audio + rtMRI learning
- Cross-attention between acoustic and articulatory representations
- Physics-informed constraints linking acoustic and articulatory dynamics
- Larger-scale self-supervised pretraining
- Domain adaptation across datasets and recording conditions
- Uncertainty-aware modeling
- Interpretable frequency-region analysis
- Joint acoustic-articulatory latent representations
- Low-latency causal variants for real-time deployment

A possible multimodal direction is:

```text
                  Speech Waveform
                       │
                       ▼
                  Audio Encoder
                       │
                       ▼
                  Audio Tokens
                       │
                       │
                       ▼
                 Cross-Attention
                       ▲
                       │
                       │
                  rtMRI Encoder
                       │
                       ▼
                ATB Representations
                       │
                       ▼
                 Shared Latent Space
                    ╱       ╲
                   ▼         ▼
          Reconstruction   Classification
```

---

# 📚 Research Paper

The complete research formulation is documented in the accompanying paper:

> **Speech-Based Parkinson’s Disease Analysis: From Acoustic Feature Engineering to Physics-Inspired Vision Transformers**

**Author:** Rishant Mallick  
**Date:** August 2026

The paper covers the complete progression from conventional speech processing and acoustic features to the physics-inspired Vision Transformer framework.

---

# 📝 Citation

If you use this work in research, please cite:

```bibtex
@article{mallick2026speech,
  title   = {Speech-Based Parkinson's Disease Analysis:
             From Acoustic Feature Engineering to Physics-Inspired Vision Transformers},
  author  = {Rishant Mallick},
  year    = {2026}
}
```

---

# 👤 Author

**Rishant Mallick**

Research focus:

- Speech Processing
- Parkinson's Disease Analysis
- Deep Learning
- Vision Transformers
- Physics-Inspired Learning
- Self-Supervised Learning
- Domain-Adversarial Learning
- Acoustic Signal Processing

---

# ⭐ Final Perspective

The framework connects classical signal processing with modern representation learning:

\[
\boxed{
\text{Speech}
\rightarrow
\text{Acoustic/Spectral Representation}
\rightarrow
\text{Learned Representation}
\rightarrow
\text{Structured Dynamics}
\rightarrow
\text{Robust Classification}
}
\]

The central objective is not simply to replace classical acoustic features with a larger neural network.

Instead, the framework combines:

**interpretability + learned representations + temporal structure + physics-inspired inductive bias + domain robustness**

to build a more structured approach to speech-based Parkinson’s disease analysis.

