# Breast Mass Risk Classification

## Overview

This project uses logistic regression to classify breast-mass samples as malignant or benign using `mean radius` and `mean texture` from the Wisconsin Diagnostic Breast Cancer dataset.

It compares:

- A NumPy logistic regression implementation
- Scikit-learn `LogisticRegression`
- A majority-class baseline

The project also examines how classification thresholds affect precision, recall, F1 score, false positives, and false negatives.

**Educational use only. This project is not intended for clinical diagnosis or medical decision-making.**

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

Two predictors are used:

1. `mean radius`
2. `mean texture`

The target is recoded as:

- `1 = malignant`
- `0 = benign`

The dataset contains:

- 569 observations
- 357 benign samples
- 212 malignant samples
- No missing values in the selected features

---

## Methods

The data is divided using stratified row indices and `random_state=42`:

- Training: 341 observations
- Validation: 114 observations
- Test: 114 observations

`StandardScaler` is fitted only on the training set and then used to transform the validation and test sets.

The NumPy logistic regression model uses:

- Full-batch gradient descent
- Mean binary cross-entropy loss
- Learning rate: `0.1`
- Epochs: `20,000`
- Initial coefficients: `0`
- No regularization
- No class weights

The NumPy model is compared with an unregularized scikit-learn logistic regression model and a `DummyClassifier(strategy="prior")` baseline.

---

## Main Findings

The NumPy training loss decreased from:

```text
0.693147
```

to:

```text
0.246834
```

The NumPy and scikit-learn models learned nearly identical coefficients:

| Parameter | NumPy | Scikit-learn |
|---|---:|---:|
| Intercept | -0.727172 | -0.727172 |
| Mean radius | 3.656794 | 3.656794 |
| Mean texture | 1.009872 | 1.009872 |

The mean absolute difference between their validation probabilities was approximately:

```text
0.0000000001
```

The validation-selected threshold was `0.60`, with an F1 score of `0.831169`.

On the test set:

| Metric | Threshold 0.50 | Threshold 0.60 |
|---|---:|---:|
| Accuracy | 0.868421 | 0.850877 |
| Precision | 0.885714 | 0.903226 |
| Recall | 0.738095 | 0.666667 |
| F1 | 0.805195 | 0.767123 |

The selected threshold improved precision but reduced recall and test F1.

The majority baseline achieved test accuracy of `0.631579` but recall and F1 of `0`, because it identified no malignant samples.

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

### Main Files

- `logistic_regression_analysis.py` — runs the complete analysis
- `memo.md` — written report and interpretation
- `reports/threshold_results.csv` — validation results for seven thresholds
- `reports/assignment_results.csv` — final validation and test results
- `reports/training_loss.png` — NumPy training-loss curve
- `requirements.txt` — required Python packages

---

# Reproducing the Project

## Prerequisites

Before running the project, make sure you have:

- Python 3.10 or newer
- Git
- Internet access for installing dependencies

Verify installation:

```text
python --version
git --version
```

On macOS, use:

```text
python3 --version
```

Python: https://www.python.org/downloads/

Git: https://git-scm.com/downloads

No separate dataset download is required.

### Opening a Terminal

**Windows:** Open the Start menu, search for **PowerShell**, and open it.

**macOS:** Open **Applications → Utilities → Terminal**.

---

## Clone the Repository

Run:

```text
git clone https://github.com/hngo13-cell/breast-mass-risk-classification
```

Then navigate into the project:

```text
cd breast-mass-risk-classification
```

Run all remaining commands from this project folder.

---

# Windows PowerShell

## 1. Create a virtual environment

```powershell
python -m venv .venv
```

## 2. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again:

```powershell
.\.venv\Scripts\Activate.ps1
```

Successful activation should show:

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

# macOS Terminal

## 1. Create a virtual environment

```bash
python3 -m venv .venv
```

## 2. Activate the virtual environment

```bash
source .venv/bin/activate
```

## 3. Install dependencies

```bash
python3 -m pip install -r requirements.txt
```

## 4. Run the analysis

```bash
python3 logistic_regression_analysis.py
```

---

## Requirements

The project directly uses:

```text
numpy
pandas
matplotlib
scikit-learn
```

These packages are installed automatically from `requirements.txt`.

---

## Tested Environment

The project was developed and tested with:

- Python 3.14
- NumPy 2.5.4
- pandas 3.1.0
- Matplotlib 3.11.2
- scikit-learn 1.9.1

Approximate runtime on the tested machine is under one minute.

Small numerical differences may occur with different supported Python or package versions.

---

## Expected Output

A successful run prints:

- Feature names and dataset dimensions
- Missing-value and class counts
- Training, validation, and test sizes
- Partition overlap checks
- NumPy training loss
- Learned coefficients
- NumPy vs. scikit-learn comparison
- Selected classification threshold
- Validation and test metrics

A successful partition check should include:

```text
Train / Validation: 0
Train / Test: 0
Validation / Test: 0
All rows covered: True
```

Training output should be close to:

```text
Initial training loss: 0.693147
Final training loss: 0.246834
```

The script creates or updates:

```text
reports/threshold_results.csv
reports/assignment_results.csv
reports/training_loss.png
```

---

## Viewing the Results

Open:

```text
reports/threshold_results.csv
```

to review validation performance across the seven candidate thresholds.

Open:

```text
reports/assignment_results.csv
```

to review the final validation and test metrics.

Open:

```text
reports/training_loss.png
```

to view the NumPy training-loss curve.

See:

```text
memo.md
```

for the complete written interpretation and research-process discussion.

---

## Troubleshooting

### `python` is not recognized

Try:

```text
python --version
```

On macOS:

```text
python3 --version
```

If Python is not found, install it and reopen the terminal.

### `ModuleNotFoundError`

Make sure the virtual environment is active and reinstall dependencies:

```text
python -m pip install -r requirements.txt
```

On macOS:

```text
python3 -m pip install -r requirements.txt
```

### PowerShell blocks virtual-environment activation

Run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then:

```powershell
.\.venv\Scripts\Activate.ps1
```

This changes the execution policy only for the current PowerShell session.

### Script cannot be found

Make sure the terminal is inside:

```text
breast-mass-risk-classification
```

Then run:

```text
python logistic_regression_analysis.py
```

### Output files are missing

Make sure the script completed without errors. The script creates the `reports` folder and writes the required files automatically.

---

## Limitations

This model is not appropriate for clinical use.

Key limitations include:

- Only two predictors are used
- The dataset is relatively small
- The dataset may not represent all populations
- Probability calibration has not been independently verified
- No external validation dataset was used
- False negatives could have serious consequences in a real clinical setting

The project is intended to demonstrate logistic regression, gradient descent, classification thresholds, and reproducible machine-learning practices.

---

## Reproducibility

The project is designed so another user can clone the repository, create a clean Python environment, install the required dependencies, run the analysis, and reproduce the output files using only the instructions in this README.

Independent peer replication is used to verify the reproducibility of the project.
