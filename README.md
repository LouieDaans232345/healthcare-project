# Stress Prediction from Physiological Signals
Final project for ML for Healthcare

Group: Biddiscombe, Daans, Kubišová, Sedra

We want a wrist-worn device to recognise stress from physiological signals. Using the WESAD dataset, we focus only on the Empatica E4 wrist sensors (BVP, EDA, TEMP, ACC) and build subject-independent models that generalise to people the model has never seen.

Research paper: https://dl.acm.org/doi/epdf/10.1145/3242969.3242985
WESAD dataset: https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html

## What we predict
- Multi-class: Baseline vs Stress vs Amusement vs Meditation
- Binary: Stress vs Non-Stress (Baseline + Amusement + Meditation)

## How we approach it
- Explore raw signals and labels, validate sampling rates, and confirm protocol timing
- Extract wrist-based features and analyze inter-subject variability
- Train multiple models (Logistic Regression, Linear SVM, XGBoost, MLP)
- Evaluate with Leave-One-Subject-Out (LOSO) cross-validation for strict subject independence

## Repository guide
- [signals.ipynb](signals.ipynb) - data preprocessing and feature extraction
- [Initial_EDA.ipynb](Initial_EDA.ipynb) — first look at raw signals, labels, and protocol
- [modeling.ipynb](modeling.ipynb) — feature matrix, LOSO training, and evaluation
- [metadata.ipynb](metadata.ipynb) — metadata and survey data extraction (extension)
- [helpers.py](helpers.py) — shared utilities
- [models](models/) — saved pipelines
- [results_loso/figs](results_loso/figs/) — evaluation figures
- [reports/metrics.json](reports/metrics.json) — stored metrics
- [stress_prediction_report.pdf](stress_prediction_report.pdf) - report

## Future work
- Add subject metadata and survey traits for personalization (see [metadata.ipynb](metadata.ipynb))
- Compare additional models such as Random Forest and AdaBoost
- Explore feature selection and robustness to motion artifacts
