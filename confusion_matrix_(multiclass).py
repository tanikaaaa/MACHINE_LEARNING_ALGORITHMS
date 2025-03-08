# -*- coding: utf-8 -*-
"""CONFUSION MATRIX (MULTICLASS).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1U6ZER4qw1wcC-yxAZp-aHoBcazLZBCRP
"""

import numpy as np
from sklearn.metrics import confusion_matrix

actual_values = np.array([0, 1, 2, 0, 1, 2, 1, 0, 2, 1, 0, 2, 1, 2, 0])
predicted_values = np.array([0, 1, 1, 0, 2, 2, 1, 0, 2, 0, 0, 2, 1, 1, 0])

categories = np.unique(actual_values)
num_classes = len(categories)
matrix = np.zeros((num_classes, num_classes), dtype=int)

for true_label, predicted_label in zip(actual_values, predicted_values):
    matrix[true_label][predicted_label] += 1

metrics_per_class = {}

for index, category in enumerate(categories):
    correct_pred = matrix[index, index]
    false_pos = np.sum(matrix[:, index]) - correct_pred
    false_neg = np.sum(matrix[index, :]) - correct_pred
    true_neg = np.sum(matrix) - (correct_pred + false_pos + false_neg)

    precision_score = correct_pred / (correct_pred + false_pos) if (correct_pred + false_pos) > 0 else 0
    recall_score = correct_pred / (correct_pred + false_neg) if (correct_pred + false_neg) > 0 else 0
    f1_measure = 2 * (precision_score * recall_score) / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0
    spec_score = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0

    metrics_per_class[category] = {
        "Precision": precision_score,
        "Recall": recall_score,
        "F1 Score": f1_measure,
        "Specificity": spec_score
    }

accuracy_overall = np.trace(matrix) / np.sum(matrix)

print("Confusion Matrix:\n", matrix)
print("\nOverall Accuracy:", accuracy_overall)
print("\nMetrics for Each Class:")
for cls, metric in metrics_per_class.items():
    print(f"\nClass {cls}:")
    for key, val in metric.items():
        print(f"{key}: {val}")