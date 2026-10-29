import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss
)


# ==================================================
# Part 2: Prepare a leakage-free experiment
# ==================================================


# --------------------------------------------------
# 1. Load the dataset, select two features, recode target
# --------------------------------------------------

data = load_breast_cancer()

feature_names = ["mean radius", "mean texture"]

feature_indices = [
    list(data.feature_names).index("mean radius"),
    list(data.feature_names).index("mean texture"),
]

X = data.data[:, feature_indices]

# Recode target:
# 1 = malignant
# 0 = benign
y = (data.target == 0).astype(int)


# --------------------------------------------------
# 2. Report feature names, missing values, class counts
# --------------------------------------------------

print("Feature names:", feature_names)
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nMissing values per feature:")
missing_counts = np.isnan(X).sum(axis=0)

for feature, count in zip(feature_names, missing_counts):
    print(f"{feature}: {count}")

print("\nOverall class counts:")
unique, counts = np.unique(y, return_counts=True)

for cls, count in zip(unique, counts):
    label = "malignant" if cls == 1 else "benign"
    print(f"Class {cls} ({label}): {count}")


# --------------------------------------------------
# 3. Split row indices
# --------------------------------------------------

indices = np.arange(len(y))

development_indices, test_indices = train_test_split(
    indices,
    test_size=0.20,
    stratify=y,
    random_state=42
)

train_indices, validation_indices = train_test_split(
    development_indices,
    test_size=0.25,
    stratify=y[development_indices],
    random_state=42
)

# Keep returned order; do not sort or shuffle again.


# --------------------------------------------------
# 4. Print partition sizes/classes and verify partitions
# --------------------------------------------------

X_train = X[train_indices]
y_train = y[train_indices]

X_validation = X[validation_indices]
y_validation = y[validation_indices]

X_test = X[test_indices]
y_test = y[test_indices]


def print_partition_info(name, indices, y_values):
    unique, counts = np.unique(y_values, return_counts=True)

    print(f"\n{name}")
    print(f"Rows: {len(indices)}")

    for cls, count in zip(unique, counts):
        label = "malignant" if cls == 1 else "benign"
        print(f"Class {cls} ({label}): {count}")


print_partition_info("Training", train_indices, y_train)
print_partition_info("Validation", validation_indices, y_validation)
print_partition_info("Test", test_indices, y_test)


print("\nOverlap checks:")
print(
    "Train / Validation:",
    len(np.intersect1d(train_indices, validation_indices))
)
print(
    "Train / Test:",
    len(np.intersect1d(train_indices, test_indices))
)
print(
    "Validation / Test:",
    len(np.intersect1d(validation_indices, test_indices))
)

all_indices = np.concatenate(
    [train_indices, validation_indices, test_indices]
)

print("\nCoverage check:")
print("Rows covered:", len(all_indices))
print("Original rows:", len(indices))
print(
    "All rows covered:",
    len(np.unique(all_indices)) == len(indices)
)


# --------------------------------------------------
# 5. Fit StandardScaler on training features only
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)
X_test_scaled = scaler.transform(X_test)

print("\nStandardScaler fitted on training data only.")

print("Scaled training shape:", X_train_scaled.shape)
print("Scaled validation shape:", X_validation_scaled.shape)
print("Scaled test shape:", X_test_scaled.shape)

# ==================================================
# Part 3: Implement NumPy logistic regression
# ==================================================


# --------------------------------------------------
# 1. Set training parameters
# --------------------------------------------------

learning_rate = 0.1
epochs = 20000


# --------------------------------------------------
# 2. Add intercept column
# --------------------------------------------------

# X_train_scaled has two columns:
# 1. mean radius
# 2. mean texture
#
# Add a leading column of ones so the model can learn
# the intercept beta_0.

X_train_aug = np.column_stack(
    [
        np.ones(X_train_scaled.shape[0]),
        X_train_scaled
    ]
)


# --------------------------------------------------
# 3. Initialize coefficients to zero
# --------------------------------------------------

# beta[0] = intercept
# beta[1] = coefficient for mean radius
# beta[2] = coefficient for mean texture

beta = np.zeros(X_train_aug.shape[1])


# --------------------------------------------------
# 4. Stable sigmoid function
# --------------------------------------------------

def sigmoid(z):
    """
    Convert logistic scores into probabilities.

    Uses a numerically stable calculation:
    exp(-logaddexp(0, -z))
    """
    return np.exp(-np.logaddexp(0, -z))


# --------------------------------------------------
# 5. Stable binary cross-entropy loss
# --------------------------------------------------

def binary_cross_entropy(z, y_true):
    """
    Mean binary cross-entropy loss.

    Uses a numerically stable form that avoids
    taking log(0).
    """
    return np.mean(
        np.logaddexp(0, z) - y_true * z
    )


# --------------------------------------------------
# 6. Record initial loss before training
# --------------------------------------------------

loss_history = []

initial_z = X_train_aug @ beta
initial_loss = binary_cross_entropy(
    initial_z,
    y_train
)

loss_history.append(initial_loss)


# --------------------------------------------------
# 7. Full-batch gradient descent
# --------------------------------------------------

n_train = len(y_train)

for epoch in range(epochs):

    # Calculate logistic score
    z = X_train_aug @ beta

    # Convert scores to probabilities
    probabilities = sigmoid(z)

    # Calculate gradient
    gradient = (
        X_train_aug.T
        @ (probabilities - y_train)
        / n_train
    )

    # Update all coefficients using the same gradient
    beta = beta - learning_rate * gradient

    # Calculate loss after the update
    updated_z = X_train_aug @ beta

    loss = binary_cross_entropy(
        updated_z,
        y_train
    )

    # Store loss for this epoch
    loss_history.append(loss)


# --------------------------------------------------
# 8. Print training results
# --------------------------------------------------

print("\nPart 3: NumPy Logistic Regression")

print(
    f"Initial training loss: "
    f"{loss_history[0]:.6f}"
)

print(
    f"Final training loss: "
    f"{loss_history[-1]:.6f}"
)

print("\nLearned coefficients (standardized units):")

print(
    f"Intercept beta_0: "
    f"{beta[0]:.6f}"
)

print(
    f"Mean radius beta_1: "
    f"{beta[1]:.6f}"
)

print(
    f"Mean texture beta_2: "
    f"{beta[2]:.6f}"
)


# --------------------------------------------------
# 9. Save training-loss plot
# --------------------------------------------------

REPORTS_DIR = (
    Path(__file__).resolve().parent
    / "reports"
)

REPORTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

plt.figure(figsize=(8, 5))

plt.plot(
    range(len(loss_history)),
    loss_history
)

plt.xlabel("Epoch")
plt.ylabel("Mean Binary Cross-Entropy Loss")

plt.title(
    "NumPy Logistic Regression Training Loss"
)

plt.tight_layout()

loss_plot_path = (
    REPORTS_DIR
    / "training_loss.png"
)

plt.savefig(
    loss_plot_path,
    dpi=150
)

plt.close()

print(
    f"\nTraining loss plot saved to: "
    f"{loss_plot_path}"
)

# ==================================================
# Part 4: Compare training implementations and baseline
# ==================================================


# --------------------------------------------------
# 1. Train scikit-learn logistic regression
# --------------------------------------------------

sklearn_model = LogisticRegression(
    C=np.inf,
    solver="lbfgs",
    max_iter=10000,
    tol=1e-9
)

sklearn_model.fit(
    X_train_scaled,
    y_train
)


# --------------------------------------------------
# 2. Train majority baseline
# --------------------------------------------------

baseline_model = DummyClassifier(
    strategy="prior"
)

baseline_model.fit(
    X_train_scaled,
    y_train
)


# --------------------------------------------------
# 3. Compare NumPy and sklearn coefficients
# --------------------------------------------------

numpy_coefficients = np.array([
    beta[0],
    beta[1],
    beta[2]
])

sklearn_coefficients = np.array([
    sklearn_model.intercept_[0],
    sklearn_model.coef_[0][0],
    sklearn_model.coef_[0][1]
])

print("\nPart 4: Model Comparison")

print("\nNumPy coefficients:")
print(f"Intercept beta_0: {numpy_coefficients[0]:.6f}")
print(f"Mean radius beta_1: {numpy_coefficients[1]:.6f}")
print(f"Mean texture beta_2: {numpy_coefficients[2]:.6f}")

print("\nScikit-learn coefficients:")
print(f"Intercept beta_0: {sklearn_coefficients[0]:.6f}")
print(f"Mean radius beta_1: {sklearn_coefficients[1]:.6f}")
print(f"Mean texture beta_2: {sklearn_coefficients[2]:.6f}")


# --------------------------------------------------
# 4. Validation probabilities
# --------------------------------------------------

# NumPy validation probabilities
X_validation_aug = np.column_stack(
    [
        np.ones(X_validation_scaled.shape[0]),
        X_validation_scaled
    ]
)

numpy_validation_z = (
    X_validation_aug @ beta
)

numpy_validation_probabilities = sigmoid(
    numpy_validation_z
)


# sklearn validation probabilities
print("\nScikit-learn classes_:")
print(sklearn_model.classes_)

class_1_index = np.where(
    sklearn_model.classes_ == 1
)[0][0]

sklearn_validation_probabilities = (
    sklearn_model.predict_proba(
        X_validation_scaled
    )[:, class_1_index]
)


# --------------------------------------------------
# 5. Mean absolute probability difference
# --------------------------------------------------

mean_absolute_probability_difference = np.mean(
    np.abs(
        numpy_validation_probabilities
        - sklearn_validation_probabilities
    )
)

print(
    "\nMean absolute difference between "
    "NumPy and sklearn validation probabilities:"
)

print(
    f"{mean_absolute_probability_difference:.10f}"
)


# --------------------------------------------------
# 6. Baseline probabilities
# --------------------------------------------------

print("\nBaseline classes_:")
print(baseline_model.classes_)

baseline_class_1_index = np.where(
    baseline_model.classes_ == 1
)[0][0]

baseline_validation_probabilities = (
    baseline_model.predict_proba(
        X_validation_scaled
    )[:, baseline_class_1_index]
)

print(
    "\nBaseline probability for class 1 "
    "(malignant):"
)

print(
    f"{baseline_validation_probabilities[0]:.6f}"
)

# ==================================================
# Part 5: Select a threshold, then evaluate
# ==================================================


# --------------------------------------------------
# 1. Evaluate candidate thresholds on NumPy validation probabilities
# --------------------------------------------------

candidate_thresholds = [
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80
]

threshold_results = []

for threshold in candidate_thresholds:

    validation_predictions = (
        numpy_validation_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_validation,
        validation_predictions,
        labels=[0, 1]
    ).ravel()

    accuracy = accuracy_score(
        y_validation,
        validation_predictions
    )

    precision = precision_score(
        y_validation,
        validation_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_validation,
        validation_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_validation,
        validation_predictions,
        zero_division=0
    )

    threshold_results.append(
        {
            "threshold": threshold,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    )


threshold_results_df = pd.DataFrame(
    threshold_results
)

print("\nPart 5: Threshold Selection")

print("\nValidation threshold results:")
print(
    threshold_results_df.to_string(
        index=False
    )
)


# --------------------------------------------------
# 2. Select threshold with highest validation F1
# --------------------------------------------------

max_f1 = threshold_results_df["f1"].max()

best_threshold_rows = threshold_results_df[
    threshold_results_df["f1"] == max_f1
]

# If tied, choose the smallest threshold
selected_threshold = (
    best_threshold_rows["threshold"].min()
)

print(
    f"\nSelected threshold: "
    f"{selected_threshold:.2f}"
)

print(
    f"Highest validation F1: "
    f"{max_f1:.6f}"
)


# --------------------------------------------------
# 3. Save threshold_results.csv
# --------------------------------------------------

threshold_results_path = (
    REPORTS_DIR
    / "threshold_results.csv"
)

threshold_results_df.to_csv(
    threshold_results_path,
    index=False,
    float_format="%.6f"
)

print(
    f"\nThreshold results saved to: "
    f"{threshold_results_path}"
)


# --------------------------------------------------
# 4. Prepare test probabilities
# --------------------------------------------------

# NumPy test probabilities
X_test_aug = np.column_stack(
    [
        np.ones(X_test_scaled.shape[0]),
        X_test_scaled
    ]
)

numpy_test_z = (
    X_test_aug @ beta
)

numpy_test_probabilities = sigmoid(
    numpy_test_z
)


# sklearn test probabilities
sklearn_test_probabilities = (
    sklearn_model.predict_proba(
        X_test_scaled
    )[:, class_1_index]
)


# baseline probabilities
baseline_test_probabilities = (
    baseline_model.predict_proba(
        X_test_scaled
    )[:, baseline_class_1_index]
)


# --------------------------------------------------
# 5. Evaluation helper function
# --------------------------------------------------

def evaluate_configuration(
    split_name,
    model_name,
    policy_name,
    threshold,
    y_true,
    probabilities
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "split": split_name,
        "model": model_name,
        "policy": policy_name,
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": accuracy_score(
            y_true,
            predictions
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "log_loss": log_loss(
            y_true,
            probabilities,
            labels=[0, 1]
        )
    }


# --------------------------------------------------
# 6. Evaluate five configurations on validation
# --------------------------------------------------

assignment_results = []

assignment_results.append(
    evaluate_configuration(
        "validation",
        "baseline",
        "default",
        0.50,
        y_validation,
        baseline_validation_probabilities
    )
)

assignment_results.append(
    evaluate_configuration(
        "validation",
        "numpy",
        "default",
        0.50,
        y_validation,
        numpy_validation_probabilities
    )
)

assignment_results.append(
    evaluate_configuration(
        "validation",
        "numpy",
        "selected",
        selected_threshold,
        y_validation,
        numpy_validation_probabilities
    )
)

assignment_results.append(
    evaluate_configuration(
        "validation",
        "sklearn",
        "default",
        0.50,
        y_validation,
        sklearn_validation_probabilities
    )
)

assignment_results.append(
    evaluate_configuration(
        "validation",
        "sklearn",
        "selected",
        selected_threshold,
        y_validation,
        sklearn_validation_probabilities
    )
)


# --------------------------------------------------
# 7. Evaluate five configurations on test
# --------------------------------------------------

assignment_results.append(
    evaluate_configuration(
        "test",
        "baseline",
        "default",
        0.50,
        y_test,
        baseline_test_probabilities
    )
)

assignment_results.append(
    evaluate_configuration(
        "test",
        "numpy",
        "default",
        0.50,
        y_test,
        numpy_test_probabilities
    )
)

assignment_results.append(
    evaluate_configuration(
        "test",
        "numpy",
        "selected",
        selected_threshold,
        y_test,
        numpy_test_probabilities
    )
)

assignment_results.append(
    evaluate_configuration(
        "test",
        "sklearn",
        "default",
        0.50,
        y_test,
        sklearn_test_probabilities
    )
)

assignment_results.append(
    evaluate_configuration(
        "test",
        "sklearn",
        "selected",
        selected_threshold,
        y_test,
        sklearn_test_probabilities
    )
)


# --------------------------------------------------
# 8. Print and save final evaluation table
# --------------------------------------------------

assignment_results_df = pd.DataFrame(
    assignment_results
)

print("\nFinal evaluation results:")

print(
    assignment_results_df.to_string(
        index=False
    )
)

assignment_results_path = (
    REPORTS_DIR
    / "assignment_results.csv"
)

assignment_results_df.to_csv(
    assignment_results_path,
    index=False,
    float_format="%.6f"
)

print(
    f"\nAssignment results saved to: "
    f"{assignment_results_path}"
)