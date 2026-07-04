## Medium Dataset Results

The medium dataset contains 150 SDSS spectra:

- 50 STAR
- 50 GALAXY
- 50 QSO

The dataset was split into 120 training spectra and 30 test spectra.

### Baseline Model

- Test accuracy: 83.3%
- Correct predictions: 25 out of 30
- PCA components: 10
- Logistic regression C: 1.0
- Macro F1-score: 0.84

### Tuned Model

- Test accuracy: 86.7%
- Correct predictions: 26 out of 30
- PCA components: 50
- Logistic regression C: 0.1
- Best cross-validation accuracy: 86.7%
- Macro F1-score: 0.86

### Comparison

Hyperparameter tuning increased test accuracy from 83.3% to 86.7%, an improvement of 3.4 percentage points. The tuned model correctly classified one additional test spectrum.

The tuned model performed best with 50 PCA components and a logistic regression C value of 0.1.

### Random Forest Model

- Test accuracy: 86.7%
- Correct predictions: 26 out of 30
- Number of trees: 500
- Macro F1-score: 0.87

### Model Comparison

| Model | Test Accuracy | Correct / 30 | Macro F1-score |
|---|---:|---:|---:|
| Baseline Logistic Regression | 83.3% | 25 / 30 | 0.84 |
| Tuned Logistic Regression | 86.7% | 26 / 30 | 0.86 |
| Random Forest | 86.7% | 26 / 30 | 0.87 |

The Random Forest matched the tuned logistic regression accuracy. It did not improve the total number of correct predictions, but it achieved a slightly higher macro F1-score.

The Random Forest made these mistakes:

- STAR → GALAXY
- QSO → GALAXY
- QSO → GALAXY
- GALAXY → QSO

The tuned logistic regression also made four mistakes, but mostly confused QSOs with STAR. This means the two models are making different kinds of errors.

### Classical Model Comparison

All classical models were evaluated on the same medium dataset split:

- Training spectra: 120
- Testing spectra: 30
- Classes: STAR, GALAXY, QSO

| Model | Test Accuracy | Correct / 30 | Mistakes | Macro F1-score |
|---|---:|---:|---:|---:|
| Baseline Logistic Regression | 83.3% | 25 / 30 | 5 | 0.836 |
| Tuned Logistic Regression | 86.7% | 26 / 30 | 4 | 0.864 |
| Random Forest | 86.7% | 26 / 30 | 4 | 0.869 |

The tuned logistic regression and Random Forest both achieved 86.7% test accuracy. Random Forest achieved the highest macro F1-score, but the improvement over tuned logistic regression was small.

This suggests that the current dataset may be too small for a more flexible classical model to clearly outperform the tuned linear baseline.