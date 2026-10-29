# Breast Mass Risk Classification: Logistic Regression and Decision Thresholds

## Part 1 — From Logistic Score to Class Decision

Logistic regression calculates a linear score, converts it into a probability using the sigmoid function, and then applies a classification threshold to determine the predicted class. In this assignment, class `1` represents malignant samples and class `0` represents benign samples.

The logistic score is `z = β₀ + β₁x`, and the probability is `p = 1 / (1 + exp(-z))`.

| Case | Intercept β₀ | Coefficient β₁ | Input x | Threshold | Score z | Probability | Predicted Class |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | -1.0 | 0.8 | 2.0 | 0.50 | 0.60 | 0.6457 | 1 |
| B | -1.0 | 0.8 | 2.0 | 0.70 | 0.60 | 0.6457 | 0 |
| C | -1.0 | -0.8 | 2.0 | 0.50 | -2.60 | 0.0691 | 0 |
| D | 1.0 | 0.0 | 2.0 | 0.50 | 1.00 | 0.7311 | 1 |

For Case A:

`z = -1.0 + (0.8 × 2.0) = 0.60`

`p = 1 / (1 + exp(-0.60)) ≈ 0.6457`

Because `0.6457 >= 0.50`, the predicted class is `1`.

Cases A and B have the same score and probability; only the threshold changes. Raising the threshold from 0.50 to 0.70 changes the final prediction from class 1 to class 0. In Case C, reversing the coefficient from positive to negative reverses the direction of the relationship between x and the predicted probability. In Case D, β₁ is zero, so x has no effect and the probability remains constant at approximately 0.7311. These coefficients were manually specified for exploration rather than learned from data.

## Part 2 — Leakage-Free Experiment Design

The Wisconsin Diagnostic Breast Cancer dataset was loaded from scikit-learn. The model used only the required predictors, `mean radius` and `mean texture`. The target was recoded so that 1 represents malignant and 0 represents benign. The dataset contained 569 observations: 357 benign and 212 malignant, with no missing values in the selected features.

The data was split using stratified row indices and `random_state=42`. The final partitions contained 341 training observations, 114 validation observations, and 114 test observations. The partitions did not overlap and together covered all 569 rows.

To prevent data leakage, `StandardScaler` was fitted only on the training features. The fitted scaler was then used to transform the training, validation, and test sets. The test partition remained untouched during model development and threshold selection.

## Part 3 — NumPy Logistic Regression

A two-feature logistic regression model was implemented directly with NumPy using full-batch gradient descent and mean binary cross-entropy loss. All coefficients were initialized to zero, with a learning rate of 0.1 and 20,000 epochs.

Training loss decreased from 0.693147 to 0.246834, and the loss curve flattened as training progressed, providing evidence of convergence. Minimizing binary cross-entropy adjusts the coefficients so that the observed labels become increasingly likely under the logistic regression model; equivalently, minimizing cross-entropy corresponds to maximizing the observed-data likelihood.

The learned coefficients were:

| Parameter | NumPy | Scikit-learn |
|---|---:|---:|
| Intercept β₀ | -0.727172 | -0.727172 |
| Mean radius β₁ | 3.656794 | 3.656794 |
| Mean texture β₂ | 1.009872 | 1.009872 |

Because the predictors were standardized, the mean-radius coefficient means that, holding mean texture fixed, a one-training-standard-deviation increase in mean radius is associated with an approximately 3.6568 increase in the log-odds of malignancy. This is an association, not a causal effect or a constant probability increase.

## Part 4 — Model and Baseline Comparison

The NumPy model was compared with an unregularized scikit-learn `LogisticRegression` model trained on the same standardized data. Their coefficients were effectively identical, and the mean absolute difference between their validation probabilities was approximately `0.0000000001`. This provides strong evidence that the NumPy implementation correctly reproduced the same unregularized logistic objective.

A `DummyClassifier(strategy="prior")` baseline estimated the malignant-class probability as 0.372434. At a threshold of 0.50, it predicted every observation as benign.

## Part 5 — Threshold Selection and Final Evaluation

Validation probabilities from the NumPy model were evaluated at thresholds from 0.20 through 0.80. The highest validation F1 score, 0.831169, occurred at a threshold of 0.60, so that threshold was selected and frozen.

At threshold 0.60, the validation confusion matrix contained 69 true negatives, 2 false positives, 11 false negatives, and 32 true positives. In plain language, the model correctly identified 32 malignant cases but incorrectly classified 11 malignant cases as benign.

On the test set, the default 0.50 threshold produced accuracy 0.868421, precision 0.885714, recall 0.738095, and F1 0.805195. Using the selected 0.60 threshold reduced false positives from 4 to 3 and increased precision to 0.903226, but false negatives increased from 11 to 14, recall decreased to 0.666667, and F1 decreased to 0.767123. The threshold was not changed after examining the test results.

The baseline achieved test accuracy of 0.631579 but recall and F1 of 0 because it identified no malignant samples. This demonstrates why accuracy alone is insufficient. Log loss remained unchanged when the threshold changed because log loss is computed from predicted probabilities, not thresholded class predictions.

## Part 6 — Limitations and Clinical Scope

This model is for educational use only and is not appropriate for clinical decision-making. It uses only two features from a relatively small historical dataset and has not been externally validated. Its probability calibration has also not been independently verified. Most importantly, false negatives could have serious consequences because a malignant case classified as benign could be missed. Therefore, the results demonstrate classification concepts rather than clinical readiness.

## Research Process

One important design decision was preventing data leakage during preprocessing. I verified that the `StandardScaler` was fitted only on the training set and then reused for validation and test data. I also checked that the three index partitions had zero overlap and together covered all 569 observations.

A second validation step was comparing my NumPy implementation with scikit-learn. The learned coefficients matched to six decimal places, and the mean absolute difference in validation probabilities was approximately `1 × 10⁻¹⁰`. This close agreement provided evidence that the manually implemented gradient-descent procedure was working correctly.