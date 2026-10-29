# Breast Mass Risk Classification

## Overview

This project implements binary logistic regression to classify breast-mass samples as malignant or benign using two features from the Wisconsin Diagnostic Breast Cancer dataset.

The project compares:

- A logistic regression model implemented manually with NumPy
- An unregularized scikit-learn `LogisticRegression` model
- A majority-class baseline using `DummyClassifier`

The analysis also evaluates how changing the classification threshold affects false positives, false negatives, precision, recall, and F1 score.

This project is for educational use only and is not intended for clinical diagnosis or medical decision-making.

---

## Research Question

Can `mean radius` and `mean texture` be used to classify breast-mass samples as malignant or benign using logistic regression?

---

## Dataset

The project uses the Wisconsin Diagnostic Breast Cancer dataset included with scikit-learn:

```python
from sklearn.datasets import load_breast_cancer
```

No external dataset download is required.

Only these two predictors are used:

1. `mean radius`
2. `mean texture`

The original scikit-learn labels are recoded so that:

- `1 = malignant`
- `0 = benign`

The dataset contains 569 observations:

- 357 benign
- 212 malignant

The selected features contain no missing values.

---

## Methods

The project follows a leakage-free train/validation/test workflow.

The data is split using stratified row indices and `random_state=42`:

- Training: 341 observations
- Validation: 114 observations
- Test: 114 observations

`StandardScaler` is fitted only on the training features. The fitted scaler is then used to transform the validation and test features.

The NumPy logistic regression model uses:

- Full-batch gradient descent
- Mean binary cross-entropy loss
- Learning rate: `0.1`
- Epochs: `20,000`
- Initial coefficients: `0`
- No regularization
- No class weights

The NumPy implementation is compared with:

```python
LogisticRegression(
    C=np.inf,
    solver="lbfgs",
    max_iter=10000,
    tol=1e-9
)
```

A majority baseline is created with:

```python
DummyClassifier(strategy="prior")
```

---

## Main Findings

The NumPy model reduced training binary cross-entropy loss from:

```text
0.693147
```

to:

```text
0.246834
```

The learned standardized-unit coefficients were:

| Parameter | NumPy | Scikit-learn |
|---|---:|---:|
| Intercept | -0.727172 | -0.727172 |
| Mean radius | 3.656794 | 3.656794 |
| Mean texture | 1.009872 | 1.009872 |

The mean absolute difference between NumPy and scikit-learn validation probabilities was approximately:

```text
0.0000000001
```

This shows that the NumPy implementation closely reproduced the scikit-learn reference model.

The best validation F1 score occurred at a classification threshold of:

```text
0.60
```

Validation F1 at threshold 0.60:

```text
0.831169
```

However, on the test set, the selected threshold did not outperform the default threshold of 0.50.

At threshold 0.50:

- Accuracy: `0.868421`
- Precision: `0.885714`
- Recall: `0.738095`
- F1: `0.805195`

At threshold 0.60:

- Accuracy: `0.850877`
- Precision: `0.903226`
- Recall: `0.666667`
- F1: `0.767123`

The majority baseline achieved test accuracy of `0.631579` but recall and F1 of `0`, because it did not identify any malignant samples.

---

## Project Structure

```text
breast-mass-risk-classification/
│
├── reports/
│   ├── assignment_results.csv
│   ├── threshold_results.csv
│   └── training_loss.png
│
├── logistic_regression_analysis.py
├── memo.md
├── README.md
├── requirements.txt
└── .gitignore
```

### Files

`logistic_regression_analysis.py`

Runs the complete analysis, including:

- Dataset loading
- Target recoding
- Train/validation/test splitting
- Standardization
- NumPy logistic regression
- scikit-learn comparison
- Baseline model
- Threshold selection
- Validation and test evaluation
- Output generation

`memo.md`

Contains the written project report, calculations, model interpretation, results, limitations, and research-process discussion.

`reports/threshold_results.csv`

Contains validation results for the required thresholds:

```text
0.20
0.30
0.40
0.50
0.60
0.70
0.80
```

`reports/assignment_results.csv`

Contains the final validation and test metrics for the baseline, NumPy, and scikit-learn models.

`reports/training_loss.png`

Shows NumPy training binary cross-entropy loss versus epoch.

---

# Reproducing the Project


## Clone the Repository

Replace the URL below with the actual GitHub repository URL if necessary:

```text
git clone https://github.com/hngo13-cell/breast-mass-risk-classification
```

Navigate into the project:

```text
cd breast-mass-risk-classification
```

All commands below should be run from the project root directory.

---

# Windows PowerShell Instructions

## 1. Create a virtual environment

```powershell
python -m venv .venv
```

## 2. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

This changes the policy only for the current PowerShell session.

A successful activation should show:

```text
(.venv)
```

at the beginning of the terminal prompt.

## 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

## 4. Run the analysis

```powershell
python .\logistic_regression_analysis.py
```

---

# macOS Terminal Instructions

## 1. Create a virtual environment

```bash
python3 -m venv .venv
```

## 2. Activate the virtual environment

```bash
source .venv/bin/activate
```

A successful activation should show:

```text
(.venv)
```

at the beginning of the terminal prompt.

## 3. Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

## 4. Run the analysis

```bash
python3 logistic_regression_analysis.py
```

---

## Expected Output

When the script runs successfully, the terminal should display:

- Feature names
- Dataset dimensions
- Missing-value counts
- Class counts
- Train/validation/test partition sizes
- Partition overlap checks
- NumPy training loss
- NumPy coefficients
- Scikit-learn coefficients
- Validation probability comparison
- Selected classification threshold
- Validation metrics
- Test metrics

The following files should also be created or updated:

```text
reports/threshold_results.csv
reports/assignment_results.csv
reports/training_loss.png
```

### Successful Partition Check

The output should show:

```text
Train / Validation: 0
Train / Test: 0
Validation / Test: 0
All rows covered: True
```

### Approximate Training Output

You should see values close to:

```text
Initial training loss: 0.693147
Final training loss: 0.246834
```

The NumPy and scikit-learn coefficients should also be nearly identical.

---

## How to View the Results

Open:

```text
reports/threshold_results.csv
```

to review the seven candidate thresholds and validation metrics.

Open:

```text
reports/assignment_results.csv
```

to review validation and test performance.

Open:

```text
reports/training_loss.png
```

to view the training-loss curve.

The written interpretation is available in:

```text
memo.md
```

---

## Troubleshooting

### `python` is not recognized

Verify whether Python is available using:

```text
python --version
```

On macOS, try:

```text
python3 --version
```

If neither command works, install Python and reopen the terminal.

---

### Missing Python module

For an error such as:

```text
ModuleNotFoundError
```

make sure the virtual environment is active, then run:

```text
python -m pip install -r requirements.txt
```

On macOS:

```text
python3 -m pip install -r requirements.txt
```

---

### PowerShell will not activate `.venv`

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then:

```powershell
.\.venv\Scripts\Activate.ps1
```

This does not permanently modify the system execution policy.

---

### Script cannot be found

Make sure the terminal is inside the project root.

You should be inside:

```text
breast-mass-risk-classification
```

Then run:

```text
python logistic_regression_analysis.py
```

---

### Output files are missing

Confirm that the script completed without errors.

The script automatically creates the `reports` directory if necessary and writes the required outputs there.

---

## Limitations

This model is not appropriate for clinical use.

Important limitations include:

- Only two predictors are used
- The dataset is relatively small
- The data may not represent all patient populations
- Probability calibration has not been independently verified
- No external validation dataset was used
- False negatives may have serious consequences in a real medical setting

The project is intended to demonstrate logistic regression, gradient descent, classification thresholds, and reproducible machine-learning workflow.

---

## Reproducibility

The project is designed so that another user can clone the repository, create a clean Python environment, install dependencies, run the analysis, and reproduce the required result files without needing access to the original development environment.

Independent peer replication is used to verify that these instructions are complete and reproducible.
