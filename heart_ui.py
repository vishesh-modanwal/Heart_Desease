 
import streamlit as st
import numpy as np
import pandas as pd
import random,time
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import joblib
import sqlite3
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import plotly.graph_objects as go
import io

 


with st.spinner("Analyzing Patient Data..."):
    time.sleep(2)

# ---------------------------------
# Page Config
# ---------------------------------
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("❤️ Heart Disease Prediction System")

st.markdown("""
 
<h4 style='text-align:center;color:gray'>
AI Powered Clinical Decision Support
</h4>
""",unsafe_allow_html=True)

st.image(
"https://images.unsplash.com/photo-1579684453423-f84349ef60b0?w=1200",
use_container_width=True
)
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
        return  FileNotFoundError

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
    
    
    
# --------------------------------- 
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
# ============================================
# Sidebar
# ============================================

with st.sidebar:
    
    # -----------------------------
    # About Project
    # -----------------------------
    with st.expander("ℹ️ About Project"):

        st.write("""
    This application predicts the probability of heart disease using a Machine Learning model.

    **Algorithm**
    - K-Nearest Neighbors (KNN)

    **Libraries**
    - Streamlit
    - Scikit-Learn
    - Plotly
    - Pandas
    - SQLite

    **Prediction**
    - Low Risk
    - High Risk

    **Output**
    - Risk Percentage
    - PDF Report
    - Patient History
    """
)

    # -----------------------------
    # Model Information
    # -----------------------------
    
    with st.expander(" Model Information"):
        st.subheader(" Model Information")

        st.metric("Model", "KNN")

        st.metric("Accuracy", "87%")

        st.metric("F1 Score", "0.86")

        st.metric("Dataset", "918 Records")

        st.metric("Features", "11")

        st.divider()

    # -----------------------------
    # Quick Statistics
    # -----------------------------
    st.subheader(" Quick Stats")

    st.success("✔ Binary Classification")

    st.info("✔ Scaled Features")

    st.info("✔ One-Hot Encoding")

    st.success("✔ Probability Prediction")

    st.divider()



 
# ---------------------------------
# Patient Information Form
# ---------------------------------
st.markdown("##  Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 1, 120, 20)
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
        st.error(f" High Risk of Heart Disease ({risk_percent:.1f}%)")
    else:
        st.success(f" Low Risk of Heart Disease ({risk_percent:.1f}%)")

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
# BMI CALCULATOR
# ---------------------------------



st.markdown("### BMI Calculator")

height=st.number_input("Height (cm)",100,220,170)

weight=st.number_input("Weight (kg)",20,200,70)

bmi=weight/((height/100)**2)

st.metric("BMI",round(bmi,2))

if bmi<18.5:
    st.info("Underweight")
elif bmi<25:
    st.success("Healthy")
elif bmi<30:
    st.warning("Overweight")
else:
    st.error("Obese")

tips=[
" Walk 30 minutes daily",
" Eat more vegetables",
" Avoid smoking",
" Exercise regularly",
" Sleep 7-8 hours"
]

st.info(random.choice(tips))



# ---------------------------------
# Patient History
# ---------------------------------
st.subheader(" Patient History")

conn = sqlite3.connect("patients.db")
history_df = pd.read_sql("SELECT * FROM history ORDER BY id DESC", conn)
conn.close()

if not history_df.empty:
    st.dataframe(history_df, use_container_width=True)


    #----------------------
    # Delete Single Record
    #----------------------

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
        
        
    #--------------------   
    # Clear All Records
    #--------------------
    if st.button("Clear All History"):
        conn = sqlite3.connect("patients.db")
        c = conn.cursor()
        c.execute("DELETE FROM history")
        conn.commit()
        conn.close()
        st.success("All history cleared.")
        st.rerun()


        
    #-------------------
    # Risk Distribution
    #-------------------

    st.subheader(" Risk Distribution")

    # Convert date column to datetime
    history_df["date"] = pd.to_datetime(history_df["date"], errors="coerce")

    # Remove invalid dates
    history_df = history_df.dropna(subset=["date"])

    # Sort by date
    history_df = history_df.sort_values("date")

    # Plotly Line Chart
    fig = px.line(
        history_df,
        x="date",
        y="risk",
        markers=True,
        title="Patient Risk Over Time"
    )

    fig.update_traces(
        line=dict(width=3),
        marker=dict(size=8)
    )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Prediction Date",
        yaxis_title="Risk (%)",
        hovermode="x unified",
        height=450,
        title_x=0.5
    )

    st.plotly_chart(fig, use_container_width=True)

    #-------------------------
    # Dashboard Metrics
    #-------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            " Patients",
            len(history_df)
        )

    with col2:
        st.metric(
            " Average Risk",
            f"{history_df['risk'].mean():.1f}%"
        )

    with col3:
        st.metric(
            " Highest Risk",
            f"{history_df['risk'].max():.1f}%"
        )

else:
    st.write("No history available")






    # Probability Chart
if "prob" in st.session_state:

    prob_df = pd.DataFrame({
        "Result": ["No Disease", "Disease"],
        "Probability": st.session_state["prob"]
    })

    st.subheader("Prediction Probability")
    st.bar_chart(prob_df.set_index("Result"))

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
 
    
    col1,  = st.columns([1])

    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
    sns.boxplot(
        x=df["HeartDisease"],
        y=df[feature],
        ax=ax
    )
    ax.set_title(f"{feature} vs Heart Disease", fontsize=12)
    st.pyplot(fig, use_container_width=False)


    # ==========================================
    # Pairplot / Distribution Analysis
    # ==========================================

    st.subheader(" Pairplot Analysis")

    # -----------------------
    # Visualization Type
    # -----------------------

    st.caption(
    "Select a visualization type and numeric columns to explore feature distributions and relationships."
    )
    
    chart_type = st.radio(
        "Select Visualization",
        ["Scatter Matrix", "Histogram", "Box Plot", "Violin Plot"],
        horizontal=True
    )


    # -----------------------
    # Color Palette
    # -----------------------

    palette_option = st.selectbox(
        "Color Palette",
        [
            "Plotly",
            "D3",
            "G10",
            "T10",
            "Alphabet",
            "Dark24",
            "Set1",
            "Pastel"
        ]
    )

    # -----------------------
    # Numeric Columns
    # -----------------------

    selected_cols = st.multiselect(
        "Select Numeric Columns",
        numeric_cols,
        default=numeric_cols[:3]
    )

    # ==========================================
    # Scatter Matrix
    # ==========================================

    if chart_type == "Scatter Matrix":

        if len(selected_cols) < 2:

            st.info("Please select at least **2 numeric columns**.")

        else:

            fig = px.scatter_matrix(
                df,
                dimensions=selected_cols,
                color="HeartDisease",
                color_discrete_sequence=getattr(
                    px.colors.qualitative,
                    palette_option
                ),
                opacity=0.75,
                width=850,
                height=600,
                template="plotly_white"
            )

            fig.update_traces(
                diagonal_visible=False,
                showupperhalf=False,
                marker=dict(size=6)
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # ==========================================
    # Histogram
    # ==========================================

    elif chart_type == "Histogram":

        if len(selected_cols) != 1:

            st.info("Please select exactly **1 numeric column**.")

        else:

            fig = px.histogram(
                df,
                x=selected_cols[0],
                color="HeartDisease",
                marginal="box",
                opacity=0.75,
                color_discrete_sequence=getattr(
                    px.colors.qualitative,
                    palette_option
                ),
                template="plotly_white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # ==========================================
    # Box Plot
    # ==========================================

    elif chart_type == "Box Plot":

        if len(selected_cols) == 0:

            st.info("Please select at least **1 numeric column**.")

        else:

            fig = px.box(
                df,
                y=selected_cols,
                color_discrete_sequence=getattr(
                    px.colors.qualitative,
                    palette_option
                ),
                points="outliers",
                template="plotly_white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # ==========================================
    # Violin Plot
    # ==========================================

    elif chart_type == "Violin Plot":

        if len(selected_cols) != 1:

            st.info("Please select exactly **1 numeric column**.")

        else:

            fig = px.violin(
                df,
                y=selected_cols[0],
                color="HeartDisease",
                box=True,
                points="all",
                color_discrete_sequence=getattr(
                    px.colors.qualitative,
                    palette_option
                ),
                template="plotly_white"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )





    corr = df[selected_cols].corr().round(2)

    st.subheader("Correlation Matrix")

    fig_corr = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )

    st.plotly_chart(fig_corr, use_container_width=True)



    st.markdown("## Describe")
    st.dataframe(
    df[selected_cols].describe().T
    )
    
    
    
except:
    st.info("heart.csv not found")
























 


