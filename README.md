Speech Dynamics: Articulatory Reconstruction and Physics-Informed Acoustic Representation Learning

A research framework for studying speech dynamics from two complementary perspectives:

Articulatory dynamics from real-time MRI (rtMRI)

Acoustic dynamics from speech spectrograms

The project combines air--tissue boundary (ATB) reconstruction, deep representation learning, a Vision Transformer (ViT), physics-inspired dynamical parameters, and domain-adversarial learning.

Overview

Speech is produced by the interaction between vocal-fold excitation and the continuously changing geometry of the vocal tract.

The central idea is:

                         SPEECH PRODUCTION
                                |
                +---------------+---------------+
                |                               |
                v                               v
          ARTICULATION                      ACOUSTICS
                |                               |
                v                               v
              rtMRI                         Waveform
                |                               |
                v                               v
        ATB contour sequence              Spectrogram
                |                               |
                v                               v
   Neighboring-frame reconstruction       ConvStem + ViT
                |                               |
                v                               v
       Reconstructed contour             h_CLS representation
                |                               |
                +---------------+---------------+
                                |
                                v
                       SPEECH DYNAMICS

rtMRI describes articulatory geometry, while the spectrogram describes the acoustic consequence of that geometry.

Research Objectives

Reconstruct missing articulatory information from neighboring rtMRI frames.

Learn representations of speech dynamics.

Model local and global time--frequency structure.

Use physics-inspired constraints to describe acoustic dynamics.

Learn frequency-dependent damping and oscillatory behavior.

Reduce dependence on recording-domain characteristics.

Exploit reconstruction/self-supervised learning when labeled data are limited.

Obtain useful representations for downstream speech classification.

Part I -- Articulatory Dynamics from rtMRI

Real-Time MRI

Real-time MRI provides a sequence of images showing the changing configuration of the vocal tract during speech.

Three air--tissue boundary contours are considered:

Contour

Region

Contour 1

Upper lip + velum

Contour 2

Lower lip + tongue base

Contour 3

Glottal/laryngeal region

Each contour is represented using 100 two-dimensional points:

[
C_t \in \mathbb{R}^{100\times2}.
]

Neighboring-Frame Reconstruction

The target contour is reconstructed from its temporal neighbors:

[
C_{t-n}, \qquad C_{t+n}.
]

The model learns:

[
\hat{C}t=f\theta(C_{t-n},C_{t+n}).
]

Conceptually:

Past frame                  Target                  Future frame
    |                          |                          |
    v                          v                          v
 C(t-n)  ----------------->  C(t)  <---------------- C(t+n)
              \                 ^                 /
               \                |                /
                +------ Model Prediction -------+

Because both past and future frames are used, this is an offline reconstruction/interpolation task, rather than strict causal forecasting.

Neighboring-Frame Baseline

A simple baseline estimates the target by averaging the neighboring contours:

\frac{C_{\mathrm{past}}+C_{\mathrm{future}}}{2}.
]

The neural network learns the nonlinear residual:

C_t-
\frac{C_{t-n}+C_{t+n}}{2}.
]

The final prediction is:

[
\hat{C}t=C{\mathrm{base}}+\hat{R}_t.
]

Arc-Length Interpolation

Raw contour annotations may contain different numbers of points and nonuniform point spacing.

For consecutive points:

[
d_i=
\sqrt{
(x_{i+1}-x_i)^2+
(y_{i+1}-y_i)^2
}.
]

The cumulative arc length is:

[
s_i=\sum_{k=1}^{i}d_k.
]

It is normalized as:

[
\hat{s}i=\frac{s_i}{s{m-1}}.
]

The contour is then resampled at uniformly spaced arc-length positions, producing a consistent:

[
100\times2
]

representation.

Models Investigated

MLP

[
(100,2)+(100,2)\rightarrow400.
]

The MLP learns global relationships between contour coordinates.

CNN

A one-dimensional CNN processes the contour sequence and captures local relationships between neighboring contour points.

Transformer

Self-attention allows relationships between distant contour points to be modeled directly.

Variational Autoencoder

[
q_\phi(z|x)=
\mathcal{N}(\mu(x),\operatorname{diag}(\sigma^2(x)))
]

with reparameterization:

[
z=\mu+\sigma\odot\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
]

GAN

A generator predicts contours while a discriminator attempts to distinguish generated contours from real contours.

Reconstruction Loss

Huber loss is:

[
L_\delta(y,\hat{y})=
\begin{cases}
\frac12(y-\hat y)^2,
& |y-\hat y|\leq\delta,\
\delta|y-\hat y|-\frac12\delta^2,
& |y-\hat y|>\delta.
\end{cases}
]

The experiments use:

[
\delta=0.1.
]

A contour smoothness term can be:

\frac{1}{N-1}
\sum_{i=2}^{N}
|\hat P_i-\hat P_{i-1}|_2^2.
]

Part II -- Acoustic Speech Representation

Speech to Spectrogram

Speech waveform
       |
       v
Preprocessing + VAD
       |
       v
Spectrogram
       |
       v
ConvStem
       |
       v
GELU
       |
       v
Stride Downsampling
       |
       v
Patch / Token Representation
       |
       v
Vision Transformer
       |
       v
h_CLS

A spectrogram represents acoustic energy as a function of time and frequency:

[
X(f,t).
]

For 50 frequency bins:

[
X_t=[X(f_1,t),\ldots,X(f_{50},t)].
]

The 50 values are frequency samples, not 50 different anatomical parts of the vocal tract.

[
\text{Articulator movement}
\rightarrow
\text{Vocal-tract configuration}
\rightarrow
\text{Acoustic response}
\rightarrow
\text{Frequency distribution}.
]

Convolutional Stem

The convolutional stem extracts local time--frequency patterns.

GELU is:

[
\operatorname{GELU}(x)=x\Phi(x).
]

Stride-based downsampling reduces resolution and therefore the number of tokens processed by the Transformer.

Vision Transformer and CLS Representation

The resulting tokens are combined with a learnable CLS token:

[
[\mathrm{CLS},P_1,\ldots,P_N].
]

After Transformer processing:

[
[\mathrm{CLS},P_1,\ldots,P_N]
\rightarrow
[h_{\mathrm{CLS}},h_1,\ldots,h_N].
]

The output corresponding to the CLS token is:

[
h_{\mathrm{CLS}}.
]

The CLS representation acts as a global summary of the input spectrogram. It aggregates information from different time--frequency regions through self-attention.

It can then be passed to a downstream classifier:

[
h_{\mathrm{CLS}}
\rightarrow
\text{Classification Head}
\rightarrow
[P(\mathrm{PD}),P(\mathrm{HC})].
]

Importantly, (h_{\mathrm{CLS}}) is the learned representation, not the final prediction.

Positional Encoding and Causality

Sinusoidal positional encoding is:

[
PE(pos,2i)=
\sin\left(
\frac{pos}{10000^{2i/d_{\mathrm{model}}}}
\right)
]

and

[
PE(pos,2i+1)=
\cos\left(
\frac{pos}{10000^{2i/d_{\mathrm{model}}}}
\right).
]

Positional encoding and causal masking are separate concepts:

Positional Encoding
        |
        +--> tells the Transformer where a token occurs

Causal Mask
        |
        +--> controls which tokens a token can attend to

For offline reconstruction, both past and future context can be used, so a non-causal Transformer is appropriate.

Part III -- Physics-Inspired Acoustic Dynamics

A damped second-order dynamical system is:

F.
]

This is a physics-inspired representation, not a claim that every spectrogram bin is literally an independent mechanical oscillator.

The parameters are interpreted as:

Parameter

Interpretation

(F)

Shared excitation/driving force

(\gamma)

Frequency-dependent damping behavior

(\omega)

Frequency-dependent oscillatory behavior

(X_f)

Acoustic response at frequency component (f)

Why (F) Has One Value While (\gamma) and (\omega) Have 50 Values

The excitation is modeled as:

[
F(t).
]

Damping and oscillatory parameters are frequency dependent:

[\gamma(f_1,t),\ldots,\gamma(f_{50},t)]
]

and

[\omega(f_1,t),\ldots,\omega(f_{50},t)].
]

The 50 values correspond to 50 acoustic frequency bins, not 50 physical vocal-tract locations.

Different acoustic frequency components can exhibit different temporal dynamics because the vocal-tract configuration shapes the acoustic energy distribution.

Dynamical Regularization

A temporal smoothness loss for the shared force is:

\sum_t|F_{t+1}-F_t|^2.
]

A gradient-difference loss is:

\sum_t
\left[
(F_{t+1}-F_t)-(F_t-F_{t-1})
\right]^2.
]

Equivalently:

\sum_t(F_{t+1}-2F_t+F_{t-1})^2.
]

The smoothness term discourages rapid frame-to-frame changes, while the gradient-difference term discourages abrupt changes in temporal slope.

Part IV -- Domain-Adversarial Learning

The encoder produces:

[
z=f_\theta(x).
]

The representation is sent to two branches:

                    Encoder
                       |
                       v
                       z
                    /     \
                   /       \
                  v         v
          Main classifier  GRL
                            |
                            v
                    Domain classifier

Gradient Reversal Layer

During the forward pass:

[
\operatorname{GRL}(z)=z.
]

During backpropagation:

-\lambda I.
]

The domain classifier tries to predict the domain correctly, while the encoder receives the reversed gradient and is encouraged to make domain-specific information less useful.

GRL does not output feature-relevance probabilities.
The domain classifier outputs domain probabilities; GRL reverses the gradient.

A conceptual objective is:

[
\min_{\theta_f,\theta_y}
\max_{\theta_d}
\left[
L_{\mathrm{class}}
-\lambda_dL_{\mathrm{domain}}
\right].
]

Reconstruction and Self-Supervised Learning

When labeled data are limited, reconstruction can help learn useful speech representations:

Input
  |
  v
Encoder
  |
  v
Latent representation
  |
  v
Reconstruction

A general self-supervised learning pipeline is:

[
\begin{aligned}
\text{Unlabeled speech}
&\rightarrow \text{SSL pretraining}\
&\rightarrow \text{Speech embedding}\
&\rightarrow \text{PD classifier}.
\end{aligned}
]

The reconstruction objective encourages the model to preserve information about speech dynamics before or alongside supervised classification.

Classical Acoustic Features

A conventional speech feature vector may contain:

[
\begin{aligned}
\Big[
&F_0,,
\sigma_{F_0},,
\mathrm{Jitter},,
\mathrm{Shimmer},,
\mathrm{HNR},,
\mathrm{RMS},,
\mathrm{ZCR},\
&F_1,,
F_2,,
F_3,,
\mathrm{MFCC}_1,,
\ldots,,
\mathrm{MFCC}_K
\Big].
\end{aligned}
]

These describe properties such as fundamental frequency, pitch variation, amplitude variation, harmonic structure, signal energy, zero-crossing behavior, formants, and cepstral characteristics.

Training and Regularization

Stochastic Depth

Stochastic depth randomly drops complete residual paths during training and acts as a regularizer for deep architectures.

Cosine Learning-Rate Schedule

[
\eta_t=
\eta_{\min}
+\frac12(\eta_{\max}-\eta_{\min})
\left[
1+\cos\left(\frac{\pi t}{T}\right)
\right].
]

Adam

[
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
]

[
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
]

After bias correction:

\theta_t-
\eta
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
]

Experimental Results

The reported average Dynamic Time Warping (DTW) results are:

Contour

Proposed Model

Baseline

Contour 1

52.41

56.74

Contour 2

90.57

106.10

Contour 3

33.69

47.90

Lower DTW indicates better contour alignment.

Absolute reductions:

[
\Delta_1=4.33,\qquad
\Delta_2=15.53,\qquad
\Delta_3=14.21.
]

Relative improvements:

Contour

Relative improvement

Contour 1

7.63%

Contour 2

14.64%

Contour 3

29.67%

Architecture Comparison

Contour

Architecture

DTW

C1

MLP

52.41

C1

CNN + MLP

51.98

C1

VAE

50.42

C1

GAN

52.87

C2

MLP

90.57

C2

VAE

93.34

C2

GAN

100.10

C3

MLP

38.52

C3

CNN + adaptive pooling + MLP

33.69

C3

CNN + MLP

36.52

C3

VAE

33.86

C3

GAN

37.81

These results suggest that different contours can benefit from different architectural inductive biases.

Dataset

The reported rtMRI data split is:

Split

Videos

Training

44

Validation

5

Testing

5

Contour representation:

Number of contours : 3
Points per contour : 100
Coordinates/point  : 2

The primary reconstruction metric is Dynamic Time Warping (DTW).

Key Conceptual Distinctions

Positional Encoding vs Causality

Positional Encoding
    |
    +--> tells the Transformer WHERE a token occurs

Causal Mask
    |
    +--> controls WHICH tokens a token can attend to

Frequency Bins vs Vocal-Tract Locations

50 frequency bins
        ≠
50 anatomical regions

GRL vs Feature Importance

GRL
 |
 +--> reverses gradient
 |
 +--> encourages domain-invariant features

GRL
 |
 X --> does NOT directly produce feature importance

(h_{\mathrm{CLS}}) vs Prediction

Spectrogram
    |
    v
ViT
    |
    v
h_CLS
    |
    v
Classifier
    |
    v
PD / HC probability

Future Work

Potential extensions include:

cross-attention between audio and rtMRI representations;

synchronized audio--rtMRI multimodal learning;

physics-informed constraints linking acoustic and articulatory representations;

uncertainty-aware contour reconstruction;

self-supervised pretraining on larger unlabeled speech datasets;

speaker and acquisition-domain adaptation;

causal/low-latency variants for real-time deployment;

interpretability of frequency regions contributing to classification;

joint learning of articulatory and acoustic latent representations.

A possible multimodal architecture is:

                 Speech waveform
                       |
                       v
                 Audio Encoder
                       |
                       v
                  Audio tokens
                       |
                       +----------------+
                                        |
                                        v
                                   Cross-Attention
                                        ^
                                        |
                       +----------------+
                       |
                       v
                  rtMRI Encoder
                       |
                       v
                ATB representations
                       |
                       v
                Shared latent space
                       |
             +---------+---------+
             |                   |
             v                   v
       Reconstruction       Classification

Repository Structure

A recommended structure is:

.
├── README.md
├── latex/
│   └── combined_speech_dynamics_research_paper.tex
├── src/
│   ├── models/
│   ├── preprocessing/
│   ├── losses/
│   └── training/
├── data/
│   └── README.md
├── experiments/
│   ├── contour1/
│   ├── contour2/
│   └── contour3/
├── results/
│   └── README.md
└── requirements.txt

Installation

git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
pip install -r requirements.txt

Typical dependencies include:

Python
PyTorch
NumPy
SciPy
scikit-learn
OpenCV
Matplotlib

Use the exact dependencies specified by the implementation when reproducing the experiments.

Running the Project

Typical workflow:

1. Prepare rtMRI / speech data
        |
        v
2. Preprocess data
        |
        v
3. Extract contours / spectrograms
        |
        v
4. Train reconstruction model
        |
        v
5. Train acoustic representation model
        |
        v
6. Apply domain-adversarial learning
        |
        v
7. Evaluate reconstruction / classification
        |
        v
8. Analyze learned speech dynamics

Example commands, if corresponding scripts exist:

python train.py
python evaluate.py

Replace these with the actual scripts in the repository.

Research Paper

The detailed research-paper version is provided in:

latex/combined_speech_dynamics_research_paper.tex

It contains the mathematical formulation, model descriptions, losses, experimental results, discussion, limitations, and references.

Citation

If this work is used in research:

@article{mallick_speech_dynamics,
  title   = {A Physics-Informed Framework for Speech Dynamics:
             Air-Tissue Boundary Reconstruction and
             Time-Frequency Representation Learning},
  author  = {Rishant Mallick},
  year    = {2026}
}

Acknowledgement

This work is associated with the Signal Processing, Interpretation and REpresentation (SPIRE) Laboratory, Indian Institute of Science (IISc), Bangalore.

License

Add the appropriate license for the repository before making the project public.
