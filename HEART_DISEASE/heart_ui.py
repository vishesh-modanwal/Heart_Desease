 





import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import sqlite3
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

 


# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("❤️ Heart Disease Prediction System")
st.write("Enter patient details to check heart disease risk")

# ---------------------------------
# Load Model Files (Cached)
# ---------------------------------
@st.cache_resource
def load_model_files():
    try:
        model = joblib.load("KNN_heart.pkl")
        scaler = joblib.load("scaler.pkl")
        columns = joblib.load("columns.pkl")
        return model, scaler, columns
    except:
        return None, None, None

model, scaler, columns = load_model_files()

if model is None:
    st.error("Model files not found. Please check .pkl files.")
    st.stop()



# ---------------------------------
# Initialize SQLite Database
# ---------------------------------
def init_db():
    conn = sqlite3.connect("patients.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age INTEGER,
            bp INTEGER,
            cholesterol INTEGER,
            maxhr INTEGER,
            oldpeak REAL,
            risk REAL,
            prediction TEXT,
            date TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()



# ---------------------------------
# Save to Database
# ---------------------------------
def save_to_db(age, bp, chol, maxhr, oldpeak, risk, prediction):
    conn = sqlite3.connect("patients.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO history (age, bp, cholesterol, maxhr, oldpeak, risk, prediction, date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        age, bp, chol, maxhr, oldpeak,
        risk,
        prediction,
        datetime.now().strftime("%Y-%m-%d %H:%M")
    ))
    conn.commit()
    conn.close()
    
    
    
    
# PDF Report
# ---------------------------------
def create_pdf(age, bp, chol, maxhr, oldpeak, risk, prediction):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    c.setFont("Helvetica", 12)
    c.drawString(100, 750, "Heart Disease Prediction Report")
    c.drawString(100, 720, f"Age: {age}")
    c.drawString(100, 700, f"Resting BP: {bp}")
    c.drawString(100, 680, f"Cholesterol: {chol}")
    c.drawString(100, 660, f"Max HR: {maxhr}")
    c.drawString(100, 640, f"Oldpeak: {oldpeak}")
    c.drawString(100, 620, f"Risk Percentage: {risk:.2f}%")
    c.drawString(100, 600, f"Prediction: {prediction}")
    c.drawString(100, 580, f"Date: {datetime.now()}")

    c.save()
    buffer.seek(0)
    return buffer



# ---------------------------------
# Sidebar - Model Info
# ---------------------------------
st.sidebar.header("Model Information")
st.sidebar.write("Model: KNN Classifier")
st.sidebar.write("Accuracy: 87%")
st.sidebar.write("F1 Score: 0.86")

# ---------------------------------
# Input Section
# ---------------------------------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 1, 120, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
    resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol", 100, 600, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])

with col2:
    rest_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise Angina", ["Y", "N"])
    oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ---------------------------------
# Prepare Input Function
# ---------------------------------
def prepare_input():
    data = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_M': int(sex == 'M'),
        'ChestPainType_ATA': int(chest_pain == 'ATA'),
        'ChestPainType_NAP': int(chest_pain == 'NAP'),
        'ChestPainType_TA': int(chest_pain == 'TA'),
        'RestingECG_Normal': int(rest_ecg == 'Normal'),
        'RestingECG_ST': int(rest_ecg == 'ST'),
        'ExerciseAngina_Y': int(exercise_angina == 'Y'),
        'ST_Slope_Flat': int(st_slope == 'Flat'),
        'ST_Slope_Up': int(st_slope == 'Up')
    }

    df = pd.DataFrame([data])

    for col in columns:
        if col not in df:
            df[col] = 0

    df = df[columns]
    scaled = scaler.transform(df)
    return df, scaled

# ---------------------------------
# Prediction Section
# ---------------------------------
if st.button("Predict"):

    input_df, input_scaled = prepare_input()

    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]
    risk_percent = prob[1] * 100
 

    pred_text = "High Risk" if prediction == 1 else "Low Risk"


    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease ({risk_percent:.1f}%)")
    else:
        st.success(f"✅ Low Risk of Heart Disease ({risk_percent:.1f}%)")

    # Risk Interpretation
    if risk_percent < 30:
        st.info("Low Risk: Maintain healthy lifestyle.")
    elif risk_percent < 60:
        st.warning("Moderate Risk: Consider medical checkup.")
    else:
        st.error("High Risk: Consult a cardiologist immediately.")

    # Save to DB
    save_to_db(age, resting_bp, cholesterol, max_hr, oldpeak, risk_percent, pred_text)

    # PDF Download
    pdf = create_pdf(age, resting_bp, cholesterol, max_hr, oldpeak, risk_percent, pred_text)
    st.download_button("Download Report", pdf, "Heart_Report.pdf", "application/pdf")


# ---------------------------------
# Patient History
# ---------------------------------
st.subheader("📋 Patient History")

conn = sqlite3.connect("patients.db")
history_df = pd.read_sql("SELECT * FROM history ORDER BY id DESC", conn)
conn.close()

if not history_df.empty:
    st.dataframe(history_df, use_container_width=True)

  # Delete Single Record
    st.markdown("### 🗑 Delete Record")
    delete_id = st.number_input("Enter ID to Delete", min_value=1, step=1)

    if st.button("Delete Selected Record"):
        conn = sqlite3.connect("patients.db")
        c = conn.cursor()
        c.execute("DELETE FROM history WHERE id = ?", (delete_id,))
        conn.commit()
        conn.close()
        st.success(f"Record with ID {delete_id} deleted successfully.")
        st.rerun()
        
        
        # Clear All Records
    if st.button("Clear All History"):
        conn = sqlite3.connect("patients.db")
        c = conn.cursor()
        c.execute("DELETE FROM history")
        conn.commit()
        conn.close()
        st.success("All history cleared.")
        st.rerun()

    # Risk Distribution
    st.subheader("Risk Distribution")
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(history_df["risk"], bins=10, kde=True, ax=ax)
    st.pyplot(fig)
else:
    st.write("No history available")






    # Probability Chart
    st.subheader("Prediction Probability")

    prob_df = pd.DataFrame({
        'Result': ['No Disease', 'Disease'],
        'Probability': prob
    })

    st.bar_chart(prob_df.set_index('Result'))

    # Patient Overview
    st.subheader("Patient Health Overview")
    st.bar_chart(input_df[['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']].T)

# ---------------------------------
# Dataset Visualization Section
# ---------------------------------
st.subheader("Dataset Visualization")


try:
    df = pd.read_csv("heart.csv")

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Boxplot Feature vs Target
    st.subheader("Feature vs Heart Disease")

    feature = st.selectbox("Select Feature", numeric_cols)

    fig, ax = plt.subplots()
    sns.boxplot(x=df["HeartDisease"], y=df[feature], ax=ax)
    ax.set_title(f"{feature} vs Heart Disease")
    st.pyplot(fig)

    # Pairplot with Color
    st.subheader("Pairplot")

    palette_option = st.selectbox(
        "Select Color Theme",
        ["deep", "muted", "bright", "pastel", "dark", "colorblind"]
    )

    selected_cols = st.multiselect(
        "Select Numeric Columns",
        numeric_cols,
        default=numeric_cols[:3]
    )

    if len(selected_cols) >= 2:
        fig = sns.pairplot(
            df[selected_cols + ["HeartDisease"]],
            hue="HeartDisease",
            palette=palette_option,
            height=5
        )
        st.pyplot(fig)

except:
    st.info("heart.csv not found")


























# @st.cache_data
# def load_dataset():
#     return pd.read_csv("heart.csv")

# try:
#     df = load_dataset()

#     st.write("Dataset Shape:", df.shape)

#     # Style Selector
#     style_option = st.selectbox(
#         "Select Plot Style",
#         ["whitegrid", "darkgrid", "white", "dark", "ticks"]
#     )
#     sns.set_style(style_option)

#     # Palette Selector
#     palette_option = st.selectbox(
#         "Select Color Theme",
#         ["deep", "muted", "bright", "pastel", "dark", "colorblind"]
#     )

#     # Hue Column
#     hue_column = st.selectbox(
#         "Select Target Column (Hue)",
#         df.columns,
#         index=df.columns.get_loc("HeartDisease") if "HeartDisease" in df.columns else 0
#     )

#     # Numeric Columns
#     numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

#     selected_cols = st.multiselect(
#         "Select Numeric Columns",
#         numeric_cols,
#         default=numeric_cols[:3]
#     )

#     if len(selected_cols) >= 2:
#         fig = sns.pairplot(
#             df[selected_cols + [hue_column]],
#             hue=hue_column,
#             palette=palette_option,
#             height=2.2,
#             diag_kind="hist"
#         )
#         st.pyplot(fig)
#     else:
#         st.warning("Select at least two numeric columns.")

# except:
#     st.info("heart.csv not found in directory.")








