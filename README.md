
# Parameter-Optimization

## Introduction
Support Vector Machines (SVMs) are a powerful machine learning algorithm widely used for classification tasks. They excel at finding hyperplanes in high-dimensional space that effectively separate data points belonging to different classes. This project explores the optimization of an SVM model for a specific classification problem.

The primary objective is to identify the best hyperparameter configuration for the SVM model using grid search. By tuning hyperparameters like regularization (C) and kernel coefficient (gamma), we aim to achieve optimal performance on unseen testing data. This approach helps us build a robust and generalizable classification model.

## Dataset
The dataset used for this project is the Letter Recognition dataset from the UCI Machine Learning Repository. The original dataset can be found at the [Letter Recognition Dataset](http://archive.ics.uci.edu/ml/datasets/Letter+Recognition) on the UCI Machine Learning Repository. It's a popular benchmark dataset for classification tasks involving handwritten character recognition.

The dataset consists of 20,000 samples, each representing a single handwritten letter. Each sample has 16 numeric features that describe various aspects of the letter image, such as the number of black pixels in specific regions. The target variable is the capital letter (A-Z) represented by the image.

This dataset provides a good example for exploring SVM model optimization due to its:

- **Moderate size:** It allows for efficient training and evaluation while offering enough data for generalization.
- **Structured data:** The features are numerical, making them suitable for direct use with SVM models.
- **Multi-class classification:** The dataset involves classifying 26 different letter classes, which is a typical scenario for SVM applications.

## Methodology
**1. Hyperparameter Tuning:**

- Grid search is employed to find the optimal hyperparameters for the SVM model. This technique involves defining a range of possible values for key hyperparameters and systematically evaluating all combinations on a validation set.
- The chosen hyperparameters for tuning are:
    - **C (regularization parameter):** Controls the trade-off between maximizing the margin (separating hyperplane) and minimizing training error. Higher C values lead to stricter separation but potentially overfitting.
    - **gamma (kernel coefficient):** Influences the influence of individual data points on the decision boundary. A higher gamma value leads to a more flexible and potentially complex decision boundary.

**2. Model Training and Evaluation:**

- The data is split into training, validation, and testing sets. The training set is used to train the SVM model with different hyperparameter combinations from the grid search.
- The validation set is used to evaluate the performance of the model on unseen data during the grid search process to avoid overfitting on the training data.
- The model with the best hyperparameters identified through grid search on the validation set is then used to train a final model on the entire training set.
- The final trained model is evaluated on the unseen testing set using performance metrics like accuracy, precision, and recall.

**Evaluation Metrics:**

- **Accuracy:** The proportion of correctly classified samples.
- **Precision:** The ratio of true positives (correctly classified positive examples) to all predicted positive examples.
- **Recall:** The ratio of true positives (correctly classified positive examples) to all actual positive examples.

These metrics provide a comprehensive understanding of the model's performance in identifying true positives and avoiding false positives/negatives.

Absolutely! Here's the content for the Results section with a comprehensive table incorporating all the information:

## Results

The hyperparameter tuning process yielded promising results for the SVM model on the Letter Recognition dataset. Here's a breakdown of the key findings:

**Grid Search and Best Parameters:**

- The grid search identified a consistent set of hyperparameters (C=10 and gamma=0.1) for the RBF kernel SVM model that achieved high training accuracy across all ten data splits. This suggests that these hyperparameter values effectively balance the trade-off between maximizing the margin and minimizing training error.

**Testing Performance:**

The following table summarizes the results for each data split:

| Sample | Best Model | Best Parameters | Best Accuracy (Training) | Testing Accuracy | Precision (weighted) | Recall (weighted) |
|---|---|---|---|---|---|---|
| 1 | SVC(C=10, gamma=0.1) | {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'} | 0.9612 | 0.9642 | 0.9648 | 0.9642 |
| 2 | SVC(C=10, gamma=0.1) | {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'} | 0.9605 | 0.9620 | 0.9626 | 0.9620 |
| 3 | SVC(C=10, gamma=0.1) | {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'} | 0.9598 | 0.9660 | 0.9666 | 0.9660 |
| 4 | SVC(C=10, gamma=0.1) | {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'} | 0.9595 | 0.9648 | 0.9654 | 0.9648 |
| 5 | SVC(C=10, gamma=0.1) | {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'} | 0.9579 | 0.9682 | 0.9684 | 0.9682 |
| 6 | SVC(C=10, gamma=0.1) | {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'} | 0.9580 | 0.9635 | 0.9638 | 0.9635 |
| 7 | SVC(C=10, gamma=0.1) | {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'} | 0.9587 | 0.9648 | 0.9653 | 0.9648 |
| 8 | SVC(C=10, gamma=0.1) | {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'} | 0.9583 | 0.9670 | 0.9672 | 0.9670 |
| 9 | SVC(C=10, gamma=0.1) | {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'} | 0.9596 | 0.9663 | 0.9668 | 0.9663 |
| 10 | SVC(C=10, gamma=0.1) | {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'} | 0.9628 | 0.9637 | 0.9642 | 0.9637 |

**Overall Observations:**

- The consistent hyperparameter selection across splits suggests that the grid search effectively identified a robust configuration for the SVM model.
- The high testing accuracy and promising values for precision and recall indicate that the model generalizes well to unseen data and performs well in terms of both identifying true positives and avoiding false classifications.

**Convergence Graph:**

![Graph](https://github.com/Barbaaryan/ParameterOptimization/blob/main/ConvergenceGraph.png?raw=true)

## Conclusion

The project successfully explored the optimization of an SVM model for a letter recognition classification task using the UCI Letter Recognition dataset. The grid search approach identified a consistent set of hyperparameters (C=10 and gamma=0.1) for the RBF kernel SVM model that achieved high training accuracy across all data splits. This demonstrates the effectiveness of the optimization process in finding a robust configuration for the model.

The model achieved strong performance on the unseen testing data, with all splits reaching accuracy above 0.96. The inclusion of precision and recall metrics alongside accuracy provides a more comprehensive understanding of the model's ability to correctly classify letters and minimize false positives/negatives. 

**Key Takeaways:**

- SVM models with appropriate hyperparameter tuning can be effective for classification tasks like letter recognition.
- Grid search offers a systematic approach to finding optimal hyperparameters for SVM models.
- Evaluating performance using metrics like accuracy, precision, and recall provides a well-rounded assessment of the model's strengths and weaknesses.
