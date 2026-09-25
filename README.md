# 🎓 Student Placement Prediction App

A clean, modern, and interactive **Streamlit web application** built for predicting student campus placement based on **CGPA** and **IQ** using a trained **Logistic Regression** machine learning model.

---

## 🌟 Key Features

1. **🎯 Instant Placement Prediction**:
   - Interactive sliders and number inputs for **CGPA** (3.0 - 10.0) and **IQ** (40 - 240).
   - Quick preset candidate profiles (*Star Performer*, *Balanced Profile*, *High IQ / Low CGPA*, *Needs Focus*).
   - Real-time placement probability meter (confidence percentage for Placed vs. Not Placed).
   - Dynamic threshold analysis showing exactly what minimum CGPA is needed to cross into the placement zone.

2. **🗺️ Interactive Decision Boundary**:
   - 2D scatter plot displaying historical student data separated by placement outcome.
   - Live **Logistic Regression Decision Boundary line** ($P = 0.5$).
   - Glowing candidate marker showing exactly where the current student stands in relation to the decision boundary.

3. **📈 Model Evaluation & Analytics**:
   - Model performance metrics on the test split: **85.0% Accuracy**, Precision, Recall, and F1-score.
   - Interactive Confusion Matrix.
   - Mathematical formulation showing the fitted Logistic Regression weights and intercept.

4. **📂 Historical Data Explorer**:
   - Filter and inspect the training dataset by placement status and minimum CGPA threshold.
   - Summary statistics (mean, std, min, max).

---

## 🚀 How to Run the App

### 1. Open Terminal / PowerShell
Navigate to this project folder:
```powershell
cd c:\Users\dell\OneDrive\Desktop\Finalproject2
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Streamlit App
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
Finalproject2/
├── app.py                   # Main Streamlit web application
├── placement-dataset.csv    # Student placement dataset (CGPA, IQ, Placement)
├── stuplac.ipynb            # Original training & EDA notebook
├── requirements.txt         # Required Python packages
└── README.md                # Documentation and setup guide
```
