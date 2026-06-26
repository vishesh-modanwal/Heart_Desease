# ❤️ Heart Disease Prediction System

An AI-powered **Heart Disease Prediction System** built with **Streamlit**, **Scikit-learn**, **SQLite**, and **ReportLab**. The application predicts the likelihood of heart disease based on patient health parameters, stores prediction history, generates downloadable PDF reports, and provides interactive data visualizations.

---

## 📌 Project Overview

Heart disease is one of the leading causes of death worldwide. Early prediction can help patients seek timely medical attention and reduce health risks.

This project uses a **Machine Learning model (K-Nearest Neighbors - KNN)** trained on the Heart Disease dataset to predict whether a patient is at risk of heart disease based on clinical attributes.

---

##  Features

###  Heart Disease Prediction
- Predicts whether a patient is at **Low Risk** or **High Risk**
- Displays prediction probability
- Provides risk interpretation

###  PDF Report
- Generates a professional PDF report
- Includes patient details
- Includes prediction result
- Includes risk percentage
- Includes prediction timestamp

###  Database Storage
- Stores every prediction in SQLite
- View prediction history
- Delete individual records
- Clear complete history

### Data Visualization
- Feature vs Heart Disease Boxplots
- Pairplot with multiple color themes
- Risk Distribution
- Interactive charts

###  User Friendly Interface
- Streamlit-based UI
- Sidebar model information
- Responsive layout
- Interactive widgets

---

#  Project Structure

```
Heart_Disease_Prediction/
│
├── heart_ui.py              # Main Streamlit Application
├── KNN_heart.pkl            # Trained KNN Model
├── scaler.pkl               # Standard Scaler
├── columns.pkl              # Feature Columns
├── heart.csv                # Dataset
├── patients.db              # SQLite Database
├── requirements.txt
├── README.md
└── assets/
```

---

#  Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| Streamlit | Web Application |
| Scikit-learn | Machine Learning |
| Pandas | Data Processing |
| NumPy | Numerical Computing |
| Matplotlib | Data Visualization |
| Seaborn | Statistical Visualization |
| SQLite | Database |
| Joblib | Model Serialization |
| ReportLab | PDF Generation |

---

#  Machine Learning Model

### Algorithm

- K-Nearest Neighbors (KNN)

### Data Preprocessing

- Missing Value Handling
- Feature Encoding
- Standard Scaling

### Input Features

| Feature |
|----------|
| Age |
| Sex |
| Chest Pain Type |
| Resting Blood Pressure |
| Cholesterol |
| Fasting Blood Sugar |
| Resting ECG |
| Maximum Heart Rate |
| Exercise Angina |
| Oldpeak |
| ST Slope |

### Target

```
HeartDisease

0 → No Disease

1 → Disease
```

---

#  Database Schema

The application automatically creates the following table.

```sql
CREATE TABLE history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER,
    bp INTEGER,
    cholesterol INTEGER,
    maxhr INTEGER,
    oldpeak REAL,
    risk REAL,
    prediction TEXT,
    date TEXT
);
```

---

#  Application Workflow

```
User Input
      │
      ▼
Feature Encoding
      │
      ▼
Standard Scaling
      │
      ▼
KNN Model Prediction
      │
      ▼
Prediction Probability
      │
      ▼
Display Result
      │
      ▼
Save to SQLite
      │
      ▼
Generate PDF Report
```

---

#  Application Screens

The application contains the following sections:

- Heart Disease Prediction
- Model Information
- Prediction Result
- Risk Interpretation
- PDF Download
- Patient History
- Delete Records
- Risk Distribution
- Dataset Visualization

---

#  Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/Heart-Disease-Prediction.git
```

Move inside project

```bash
cd Heart-Disease-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Streamlit

```bash
streamlit run heart_ui.py
```

---

#  Requirements

```
streamlit
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
reportlab
plotly
```

Install manually

```bash
pip install streamlit numpy pandas matplotlib seaborn scikit-learn joblib reportlab plotly
```

---

#  Visualizations Included

- Feature Distribution
- Feature vs Heart Disease
- Pairplot
- Risk Distribution
- Prediction Probability
- Patient Overview

---

#  PDF Report Includes

- Patient Age
- Blood Pressure
- Cholesterol
- Maximum Heart Rate
- Oldpeak
- Risk Percentage
- Prediction
- Date & Time

---

#  Database Operations

- Save Predictions
- View History
- Delete Single Record
- Clear Entire History

---

#  Future Improvements

- Random Forest / XGBoost comparison
- Multiple ML models
- Hyperparameter tuning
- SHAP Explainability
- User Authentication
- Cloud Deployment
- Dark Mode
- Doctor Dashboard
- Email PDF Reports
- REST API Integration

---

#  Model Performance

| Metric | Value |
|---------|-------|
| Algorithm | KNN |
| Accuracy | 87% |
| F1 Score | 0.86 |

> *Performance values are based on the trained model used in this project.*

---

# Learning Outcomes

This project demonstrates:

- Machine Learning Model Deployment
- Data Preprocessing
- Feature Engineering
- Model Serialization
- Streamlit Development
- SQLite Integration
- Report Generation
- Data Visualization
- End-to-End ML Application Development

---

#  Author

**Suraj**

Aspiring Data Scientist | Machine Learning Enthusiast

---

#  License

This project is intended for educational and portfolio purposes.

---

## If you found this project useful, consider giving it a Star on GitHub!