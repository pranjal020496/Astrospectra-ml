# Model Card: AstroSpectra ML Classifiers

## Intended Use

This project demonstrates a reproducible machine-learning workflow for SDSS astronomical spectra. It is intended for research-software demonstration, exploratory analysis, and educational use.

It should not be used as an authoritative astronomical classification service.

## Classes

The models classify spectra into:

- STAR
- GALAXY
- QSO

## Dataset

The current medium dataset contains 150 SDSS spectra:

| Class | Count |
|---|---:|
| STAR | 50 |
| GALAXY | 50 |
| QSO | 50 |

Each spectrum is preprocessed to 2048 standardized flux values across the approximate wavelength range 3800–9000 Å.

## Models

| Model | Test Accuracy | Macro F1 |
|---|---:|---:|
| Baseline Logistic Regression | 83.3% | 0.836 |
| Tuned Logistic Regression | 86.7% | 0.864 |
| Random Forest | 86.7% | 0.869 |
| 1D CNN | 73.3% | 0.730 |

## Main Findings

The tuned logistic regression and Random Forest models performed best on the current dataset.

The CNN underperformed, likely because the dataset is small for deep learning.

The anomaly detector found that the most unusual spectra were mostly QSOs. Some of these spectra were also misclassified, suggesting that unusual QSO spectra are a key challenge in this dataset.

## Limitations

- The dataset is small.
- The spectra are not a complete or representative SDSS sample.
- Raw FITS files and trained model artifacts are not committed to GitHub.
- The CNN is undertrained relative to what a deep-learning model would need.
- Predictions are intended for workflow demonstration, not scientific classification.
- Larger, more diverse datasets would be needed for robust research use.

## Reproducibility

The project uses documented scripts, fixed random seeds, manifest files, and separated raw data, processed data, model artifacts, and output reports.

## Ethical and Scientific Use

Users should treat the app as an exploratory research-software prototype. Any scientific conclusion should be checked against validated astronomical catalogues and domain-specific analysis.