# Stress Prediction from Physiological Signals
## Final project of ML&DL for Healthcare
#### Group: Biddiscombe, Daans, Kubišová, Sedra

Idea of project: Follow research paper "Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection" and recreate the experiment.

**Research paper link:** https://dl.acm.org/doi/epdf/10.1145/3242969.3242985

**WESAD dataset:** https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html

**Note:** Only use data from wrist device (Empatica E4). 

<hr>

**Objectives:**
Develop ML moddels to predict stress conditions from physiological signals collected from a wearable wrist device. 
The task can be formulated as a binary classification problem with classes "Stress" and "Non-stress", or as a multi-class problem, with classes like "Baseline", "Stress", "Amusement".
The focus is on understanding how physiological signals measured at the wrist can be used to detect affective states.

You must use different models than those proposed in the original paper.

<hr>

### Methodology:
- **Data Exploration and Preprocessing:** Load and inspect physiological signals (e.g., BVP, EDA, temperature). Select only data acquired from the wrist-worn device (MUST) (Empatica E4). Apply preprocessing (e.g., filtering, normalization, artifact removal). Segment the data into time windows suitable for analysis.
- **Feature Extraction / Representation:** Extract relevant features from physiological signals (e.g., statistical, temporal, and frequency-based features). Optionally use established libraries (e.g., FLIRT, NeuroKit) or develop a custom feature extraction pipeline Analyze feature relevance and variability across subjects.
- **Model Development:** Implement classification models using approaches different from those proposed in the original study Compare multiple methods (e.g., logistic regression, support vector machines, tree-based models, or neural networks). Explore both binary and multi-class formulations of the problem
- **Validation strategy:** MUST Implement a leave-one-subject-out cross-validation scheme to ensure subject-independent evaluation. Analyze how model performance generalizes across individuals.
- **Model evaluation**: Evaluate performance using metrics such as accuracy, confusion matrix, precision, recall, F1-score, and ROC curves. Compare models and analyze strengths and limitations.
- **Extension (optional):** Investigate personalization vs. general models Explore feature selection or dimensionality reduction Study robustness to noise and inter-subject variability.
