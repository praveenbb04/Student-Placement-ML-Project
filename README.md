# 🎓 Student Placement Prediction Using Machine Learning

> **Presentation + GitHub Markdown**
>
> This project uses **Logistic Regression** to predict whether a student is likely to be placed based on **CGPA** and **IQ score**. The model is integrated into an interactive Streamlit application.

---

# Slide 1 — Title

## 🎓 Student Placement Prediction Using Machine Learning

### Predicting Student Campus Placement with Logistic Regression

**Technology Stack**
- 🐍 Python
- 📊 Pandas & NumPy
- 🤖 Scikit-learn
- 📈 Plotly / Matplotlib / Seaborn
- 🌐 Streamlit
- 📓 Jupyter Notebook

**Input Features**
- CGPA
- IQ Score

<img width="1873" height="906" alt="Screenshot 2026-09-25 113848" src="https://github.com/user-attachments/assets/33bc5679-b43c-4178-b875-49c90cc27e60" />

**Output**
- Placed / Not Placed
- Placement probability

---

# Slide 2 — Objective and Implementation Steps

## 🎯 Objective

The objective of this project is to develop a machine learning model that predicts whether a student will be placed based on their **CGPA and IQ score**.

The project uses a **Logistic Regression classification model** trained on a dataset containing 100 student records.

## ⚙️ Implementation Steps

### 1. Data Collection
- Load the `placement-dataset.csv` dataset.
- Dataset contains **100 records**.
- Features: `cgpa`, `iq`
- Target: `placement`

### 2. Data Preprocessing
- Remove the unnecessary `Unnamed: 0` column.
- Check the dataset for missing values.
- Separate input features (`X`) and target (`y`).

### 3. Train-Test Split
- Split the data into:
  - **80% training data**
  - **20% testing data**
- Use `random_state=42` and stratification.

### 4. Model Training
- Apply **Logistic Regression**.
- Train the model using CGPA and IQ.

### 5. Model Evaluation
- Predict the test data.
- Calculate:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

### 6. Prediction
- Enter a student's CGPA and IQ.
- Predict whether the student is likely to be placed.
- Display the probability of placement.

### 7. Web Application
- Deploy the trained model through a Streamlit interface.
- Provide interactive student inputs, prediction probability, decision boundary visualization, model metrics, and dataset exploration.

**Project flow:**

```text
Dataset
   ↓
Data Cleaning
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Logistic Regression
   ↓
Model Evaluation
   ↓
Student Input
   ↓
Placement Prediction
```

---

# Slide 3 — Code Screenshot

## 💻 Important Code Sections

### Import Required Libraries

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
```

### Select Features and Target

```python
X = data[["cgpa", "iq"]]
y = data["placement"]
```

### Split the Dataset

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Train the Logistic Regression Model

```python
model = LogisticRegression()

model.fit(X_train, y_train)
```

### Make a Prediction

```python
new_student = pd.DataFrame({
    "cgpa": [8.0],
    "iq": [120]
})

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("Prediction: Student is likely to be PLACED 🎉")
else:
    print("Prediction: Student is likely to be NOT PLACED.")
```

### Calculate Probability

```python
probability = model.predict_proba(new_student)

print(
    "Probability of PLACED:",
    round(probability[0][1] * 100, 2),
    "%"
)
```

## 📸 Code Screenshot

> **Insert your Jupyter Notebook / VS Code screenshot here.**

```text
📷 Screenshot:
   stuplac.ipynb — Model Training Code
```

The original notebook contains the complete workflow from importing libraries and loading the dataset to model training, evaluation, confusion matrix generation, and prediction.

---

# Slide 4 — Output Screenshots

## 📊 Model Output

### Model Performance

| Metric | Result |
|---|---:|
| Dataset Size | 100 records |
| Training Data | 80 records |
| Testing Data | 20 records |
| Algorithm | Logistic Regression |
| Test Accuracy | **85.0%** |
| Placed Precision | **88.89%** |
| Placed Recall | **80.00%** |
| Placed F1-Score | **84.21%** |

### Confusion Matrix

```text
                 Predicted
              Not Placed   Placed

Actual
Not Placed         9          1
Placed             2          8
```

### Sample Prediction

For the notebook's sample student:

```text
CGPA = 8.0
IQ   = 120

Prediction: Student is likely to be PLACED 🎉
```

The calculated probability for this sample is approximately:

```text
Not Placed : 0.26%
Placed     : 99.74%
```

## 🖥️ Streamlit Application Screenshots

### Screenshot 1 — Student Prediction Dashboard

> **Insert screenshot of the Streamlit prediction page here.**

```text
📷 Screenshot:
   Student Placement Predictor — Prediction & Insights
```

### Screenshot 2 — Decision Boundary

> **Insert screenshot of the CGPA vs IQ decision-boundary visualization here.**

```text
📷 Screenshot:
   Logistic Regression Decision Boundary
```

### Screenshot 3 — Model Evaluation

> **Insert screenshot showing accuracy, precision, recall, F1-score and confusion matrix here.**

```text
📷 Screenshot:
   Model Evaluation & Performance
```

### Screenshot 4 — Dataset Explorer

> **Insert screenshot of the dataset explorer and filtering section here.**

```text
📷 Screenshot:
   Historical Placement Dataset Explorer
```

---

# Slide 5 — Application or Uses

## 🚀 Applications

### 1. 🎓 Student Placement Analysis
Can be used as a basic predictive tool to estimate placement outcomes from academic and cognitive features.

### 2. 📊 Academic Performance Analysis
Can help visualize the relationship between **CGPA, IQ, and placement outcomes** in the provided dataset.

### 3. 🏫 College Placement Support
Can be adapted as a prototype for placement cells to explore historical placement data.

### 4. 🧪 Machine Learning Education
Useful for learning:
- Classification
- Logistic Regression
- Train-test splitting
- Model evaluation
- Confusion matrices
- Prediction probabilities

### 5. 🌐 Interactive ML Applications
The Streamlit application demonstrates how a trained ML model can be converted into an interactive web application.

### 6. 📈 Data Visualization
The application provides:
- Student distribution
- Decision boundary
- Placement probabilities
- Confusion matrix
- Dataset statistics

> **Important:** This is a machine-learning prototype based only on CGPA and IQ in the supplied dataset. Real-world placement decisions involve many additional factors such as skills, communication, internships, projects, interview performance, and job requirements.

---

# Slide 6 — Conclusion and References

## ✅ Conclusion

The **Student Placement Prediction** project demonstrates how machine learning can be applied to a classification problem using student-related data.

A **Logistic Regression** model was trained using **CGPA and IQ** as input features. The model was evaluated using an 80/20 stratified train-test split and achieved **85% test accuracy** on the supplied dataset.

The project was further developed into an interactive **Streamlit web application**, allowing users to enter student information, view placement probabilities, explore the decision boundary, inspect model metrics, and explore the dataset.

### Key Learning Outcomes

- Understanding a machine learning classification workflow
- Data preprocessing and feature selection
- Train-test splitting
- Logistic Regression
- Prediction probabilities
- Accuracy, precision, recall and F1-score
- Confusion matrix
- Data visualization
- Building an ML web application using Streamlit

## 📚 References

1. **Project Dataset**
   - `placement-dataset.csv`
   - 100 student records with CGPA, IQ and placement labels.

2. **Project Notebook**
   - `stuplac.ipynb`
   - Contains data preprocessing, exploratory visualization, model training, evaluation and prediction.

3. **Streamlit Application**
   - `app.py`
   - Interactive Student Placement Predictor web application.

4. **Python Dependencies**
   - `requirements.txt`
   - Contains Streamlit, Pandas, NumPy, Scikit-learn, Plotly, Matplotlib and Seaborn dependencies.

5. **Scikit-learn**
   - Logistic Regression and model evaluation utilities used in the project.

---

## 📁 GitHub Project Structure

```text
Student-Placement-Prediction/
│
├── app.py
├── placement-dataset.csv
├── stuplac.ipynb
├── requirements.txt
└── README.md
```

## ▶️ How to Run

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```

The application will open in the browser at:

```text
http://localhost:8501
```

---

## ⭐ Project Summary

**Student Placement Prediction** is a beginner-friendly machine learning project that combines **Logistic Regression + Data Visualization + Streamlit** to create an interactive student placement prediction system.

> **Built with Python, Scikit-learn and Streamlit.**
