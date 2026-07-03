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