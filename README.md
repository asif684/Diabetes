# README: Diabetes Prediction Using Logistic Regression

This project demonstrates a basic implementation of logistic regression to predict the likelihood of diabetes based on various health metrics. The code is designed to run on Google Colab and uses Python libraries such as Pandas and Scikit-learn.

---

## **Project Overview**
The dataset used in this project contains health-related data for individuals and includes the following features:
- **Pregnancies**: Number of pregnancies.
- **Glucose**: Plasma glucose concentration.
- **Diastolic**: Diastolic blood pressure (mm Hg).
- **Triceps**: Triceps skinfold thickness (mm).
- **Insulin**: 2-Hour serum insulin (mu U/ml).
- **BMI**: Body mass index.
- **DPF (Diabetes Pedigree Function)**: A function that scores the likelihood of diabetes based on family history.
- **Age**: Age in years.
- **Diabetes**: Binary target variable indicating diabetes diagnosis (1 = diabetes, 0 = no diabetes).

The pipeline includes data import, preprocessing, splitting into training and test datasets, model training, and evaluation.

---

## **Steps in the Code**

### 1. **Import Libraries**
   - `pandas` for data manipulation.
   - `sklearn` for machine learning tasks.

### 2. **Import Data**
   - The diabetes dataset is loaded from a GitHub repository.
   - Basic exploratory data analysis is performed using `.info()`, `.describe()`, and `.head()`.

### 3. **Define Features (X) and Target (y)**
   - **X**: Predictor variables (health metrics).
   - **y**: Target variable (diabetes diagnosis).

### 4. **Split the Data**
   - The dataset is split into training and testing subsets using `train_test_split` from Scikit-learn.

### 5. **Select a Model**
   - Logistic Regression is chosen for its simplicity and effectiveness in binary classification tasks.

### 6. **Train the Model**
   - The model is trained using the training data (`X_train`, `y_train`).

### 7. **Predict**
   - Predictions are made on the test dataset (`X_test`).

### 8. **Evaluate Accuracy**
   - Accuracy is calculated using `accuracy_score` from Scikit-learn.
   - **Achieved Accuracy**: 77.6%

---

## **Setup Instructions**
1. **Environment**: Ensure you are running the code in a Python 3 environment, preferably Google Colab.
2. **Dependencies**: Install the following Python libraries if not already available:
   ```bash
   pip install pandas scikit-learn
   ```
3. **Run the Code**: Copy the code snippet into a Colab notebook and execute the steps sequentially.

---

## **File Descriptions**
- **Diabetes.csv**: Dataset sourced from the YBI Foundation's GitHub repository.

---

## **Future Improvements**
- Tune hyperparameters for the Logistic Regression model.
- Explore other classification algorithms (e.g., Random Forest, SVM).
- Perform feature engineering to improve accuracy.
- Incorporate cross-validation for robust evaluation.

---

## **Contact**
For any inquiries or suggestions, feel free to reach out to **Mohammed Asif Ameen Baig**.
