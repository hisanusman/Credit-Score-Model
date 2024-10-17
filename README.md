# Machine Learning Model for Credit Score Classification

This repository contains a Jupyter notebook demonstrating a machine learning workflow using scikit-learn to classify credit scores into three categories: "Good", "Standard", and "Poor". The objective is to build a robust classification model that accurately predicts the credit score category of individuals based on their financial behavior and related features. The entire workflow includes data preprocessing, model development, hyperparameter tuning, evaluation, and model interpretation.

## Dataset Description

The dataset used in this project contains information about individuals' financial behavior, including features like the number of bank accounts, credit mix, payment behavior, and more. The target variable, `Credit_Score_Encoded`, categorizes credit scores into three distinct classes:

- **Good**
- **Standard**
- **Poor**

### Key Features in the Dataset:
- `Num_Bank_Accounts`: Number of bank accounts held by the individual.
- `Occupation`: Type of occupation the individual is involved in.
- `Credit_Mix`: The mix of credit types (e.g., credit cards, loans).
- `Payment_of_Min_Amount`: Whether the individual pays the minimum amount due.
- `Payment_Behaviour`: Pattern of payment behavior exhibited by the individual.
- **Many more financial and behavioral attributes**.

## Workflow

### 1. **Data Preprocessing**
   Handling missing values and cleaning data are critical steps to ensure the accuracy and performance of the machine learning model.

   **Handling Missing Values:**
   - **Categorical Features**: For categorical columns such as `Occupation`, `Credit_Mix`, and `Payment_of_Min_Amount`, missing values were handled using the most frequent value (mode) imputation. This approach was chosen to preserve the distribution of these categorical features while minimizing information loss.
   - **Numerical Features**: For numerical columns like `Num_Bank_Accounts`, `Interest_Rate`, and `Delay_from_due_date`, missing values were imputed using the median. Median imputation was selected as it is robust to outliers and ensures that the central tendency of the data is maintained without being influenced by extreme values.
   - **Outlier Handling**: Outliers were detected using techniques such as the interquartile range (IQR) method, where extreme values were identified and addressed. Rows containing extremely large values were either capped or removed based on their impact on model performance. In particular, columns containing infinite or very large values were dropped to prevent model training errors.

   **Categorical Encoding:**
   - **One-Hot Encoding**: For nominal categorical variables like `Occupation` and `Credit_Mix`, one-hot encoding was applied. This approach ensured that the model could treat each category as a separate binary feature without assuming any ordinal relationship.
   - **Label Encoding**: Ordinal categorical variables, such as `Credit_Score`, were label-encoded to preserve the inherent order in the data. For instance, "Good", "Standard", and "Poor" were encoded numerically to represent the credit score hierarchy.

   **Feature Scaling and Normalization**: 
   - Numerical features were standardized using the `StandardScaler` to ensure that the data was on a similar scale. Standardization helps with faster model convergence, particularly for distance-based algorithms, and ensures fair treatment of all numerical variables.
   
### 2. **Feature Selection**
   To enhance model performance, careful selection of features was carried out:

   **High Cardinality Features**: Columns with too many unique categories (e.g., `Payment_Behaviour`) were transformed using target encoding or reduced to manageable categories based on domain knowledge. This reduced the risk of overfitting while preserving the most informative categories.
   
   **Correlation Analysis**: We performed correlation analysis to identify highly correlated features, which were then either dropped or combined to avoid multicollinearity. This process helped eliminate redundant information, which can confuse the model and lead to poor generalization.

   **Variance Thresholding**: Low variance features that contributed little to the prediction task were removed. These were features that exhibited nearly constant values across samples and provided little information gain.

   **Domain Expertise**: Features directly related to financial behavior, such as `Payment_of_Min_Amount`, `Credit_Mix`, and `Delay_from_due_date`, were retained as they are highly predictive of an individual's creditworthiness. Features like `Num_Bank_Accounts` and `Occupation` were also included based on their influence on credit behavior.

### 3. **Model Development**
   - **Random Forest Classifier**: Implemented using a simple classifier without grid search for parameter tuning. Evaluated performance based on accuracy and classification report.
   - **XGBoost Classifier**: Implemented to improve performance using important hyperparameters like `max_depth`, `learning_rate`, and `n_estimators`. Aimed to balance model complexity and overfitting.

### 4. **Model Evaluation**
   - **Accuracy**: Evaluated on both training and testing data to measure model performance.
   - **Classification Report**: Detailed evaluation using metrics like precision, recall, and F1-score for all three classes.
   - Focused on identifying overfitting issues and adjusted parameters to improve test accuracy without sacrificing training performance.

### 5. **Class Balancing**
   - The dataset contained unbalanced classes for the target variable, and class balancing techniques were applied (SMOTE) to improve model generalization and handle the disparity in class distribution.

## Results

- **Random Forest**: Initially used for baseline performance. Hyperparameters were tweaked manually for slight improvements but were constrained by class imbalance.
- **XGBoost**: Improved model accuracy on both training and test datasets, though with a trade-off in training speed due to the complexity of the model.
  
**Final Model Accuracy**: The best model achieved an accuracy of over 85% on test data with balanced classes, and classification reports showed substantial improvements in predicting all three credit score classes.

## Usage

To run the notebook:
1. Clone the repository.
2. Install required libraries using `pip install -r requirements.txt`.
3. Run the Jupyter notebook `Credit_Score_Classification.ipynb`.

## Conclusion

This project demonstrates the complete process of building a machine learning model using scikit-learn, from data preprocessing to model deployment. The models built show promise for predicting credit scores based on various financial and behavioral features, making it a valuable tool for credit risk analysis and financial planning.

(Please note that this modal was developed during my internship at Teradata and is to be used or implemented under the supervision of a data scientist. Neither I, nor Teradata, take the responsibilty of any wrong predictions that the modal might make).
