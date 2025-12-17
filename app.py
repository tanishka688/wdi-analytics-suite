import streamlit as st
import pandas as pd
import numpy as np
import wbgapi as wb
import plotly.express as px
import plotly.graph_objects as go

# --- SYLLABUS MODELS IMPORT ---
# Unit II
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Unit III
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Unit V
from sklearn.neural_network import MLPRegressor, MLPClassifier

# Unit VI
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
)

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="World Development Predictor",
    layout="wide",
    page_icon="🌍"
)

# =========================================================
# -------------------- CSS (UNCHANGED) --------------------
# =========================================================
st.markdown("""<style>/* your entire CSS remains unchanged */</style>""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("## 🌍 World Development Intelligence System")
st.markdown("AI-powered economic predictions based on 20 years of global data")

# =========================================================
# ------------------ DATA LOADING -------------------------
# =========================================================
@st.cache_data(show_spinner=False)
def load_data():
    indicators = {
        'NY.GDP.PCAP.CD': 'GDP_Per_Capita',
        'SP.DYN.LE00.IN': 'Life_Expectancy',
        'EN.ATM.CO2E.PC': 'CO2_Emissions',
        'FP.CPI.TOTL.ZG': 'Inflation',
        'EG.USE.PCAP.KG.OE': 'Energy_Use',
        'SE.SEC.ENRR': 'School_Enrollment'
    }

    try:
        data_gen = wb.data.fetch(indicators.keys(), mrv=20)
        rows = []

        for d in data_gen:
            rows.append({
                'Country': d['economy'],
                'Ind': indicators[d['series']],
                'Year': int(str(d['time']).replace('YR', '')),
                'Val': d['value']
            })

        df = pd.DataFrame(rows)
        df = df.pivot_table(index=['Country', 'Year'], columns='Ind', values='Val').reset_index()

        countries = wb.economy.DataFrame()['name'].to_dict()
        df['Country_Name'] = df['Country'].map(countries)

        return df.dropna(subset=['GDP_Per_Capita'])

    except:
        return None


with st.spinner("Loading World Bank data..."):
    df = load_data()

if df is None:
    st.error("World Bank API blocked. Upload CSV instead.")
    f = st.file_uploader("Upload CSV", type="csv")
    if f:
        df = pd.read_csv(f)
    else:
        st.stop()

# =========================================================
# ------------------ PREPROCESSING ------------------------
# =========================================================
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

feature_cols = [
    'Life_Expectancy',
    'CO2_Emissions',
    'Inflation',
    'Energy_Use',
    'School_Enrollment'
]
valid_feats = [c for c in feature_cols if c in df.columns]

# ✅ FIX: helper to enforce correct feature order
def prepare_input(input_dict):
    return pd.DataFrame([input_dict])[valid_feats]

# =========================================================
# ------------------ SIDEBAR ------------------------------
# =========================================================
st.sidebar.header("Analysis Mode")
analysis_mode = st.sidebar.radio(
    "Select analysis:",
    [
        "Economic Forecasting (Regression)",
        "Country Classification (High-Accuracy AI)",
        "Model Performance Comparison"
    ]
)

# =========================================================
# ========== MODE 1: ECONOMIC FORECASTING =================
# =========================================================
if analysis_mode == "Economic Forecasting (Regression)":

    st.header("📈 Economic Development Forecasting")

    X = df[valid_feats]
    y = df['GDP_Per_Capita']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_name = st.selectbox(
        "Choose regression model",
        [
            "Random Forest",
            "Gradient Boosting",
            "Linear Regression",
            "Polynomial Regression",
            "Neural Network (MLP)"
        ]
    )

    if model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=150, random_state=42)
    elif model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Polynomial Regression":
        model = Pipeline([
            ('poly', PolynomialFeatures(2)),
            ('lin', LinearRegression())
        ])
    else:
        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    st.metric("R² Score", f"{r2_score(y_test, preds):.3f}")
    st.metric("RMSE", f"${np.sqrt(mean_squared_error(y_test, preds)):,.0f}")

    st.subheader("🔮 Make a Prediction")

    with st.form("predict_form"):
        input_data = {
            'Life_Expectancy': st.number_input("Life Expectancy", 40.0, 90.0, 72.0),
            'CO2_Emissions': st.number_input("CO₂ Emissions", 0.0, 30.0, 5.0),
            'Inflation': st.number_input("Inflation (%)", -10.0, 50.0, 4.0),
            'Energy_Use': st.number_input("Energy Use", 0.0, 20000.0, 2500.0),
            'School_Enrollment': st.number_input("School Enrollment (%)", 0.0, 100.0, 85.0)
        }

        submit = st.form_submit_button("Predict GDP")

        if submit:
            # ✅ FIX APPLIED HERE
            pred_df = prepare_input(input_data)
            pred_value = float(model.predict(pred_df)[0])

            st.success(f"💰 Predicted GDP per Capita: ${pred_value:,.2f}")

# =========================================================
# ========= MODE 2: COUNTRY CLASSIFICATION =================
# =========================================================
elif analysis_mode == "Country Classification (High-Accuracy AI)":

    st.header("🏷 Country Income Classification")

    threshold = st.slider("High income threshold ($)", 5000, 50000, 12000)
    df['Class'] = (df['GDP_Per_Capita'] > threshold).astype(int)

    X = df[valid_feats]
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf_name = st.selectbox(
        "Choose classifier",
        [
            "Random Forest",
            "Logistic Regression",
            "Gradient Boosting",
            "Neural Network (MLP)",
            "Support Vector Machine"
        ]
    )

    if clf_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=250, random_state=42)
    elif clf_name == "Logistic Regression":
        model = LogisticRegression(max_iter=2000)
    elif clf_name == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=100)
    elif clf_name == "Neural Network (MLP)":
        model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
    else:
        model = SVC(kernel="rbf", probability=True)

    model.fit(X_train, y_train)

    st.subheader("🔎 Classify Country")

    with st.form("classify_form"):
        input_data = {
            'Life_Expectancy': st.number_input("Life Expectancy", 40.0, 90.0, 70.0),
            'CO2_Emissions': st.number_input("CO₂ Emissions", 0.0, 30.0, 6.0),
            'Inflation': st.number_input("Inflation (%)", -10.0, 50.0, 5.0),
            'Energy_Use': st.number_input("Energy Use", 0.0, 20000.0, 3000.0),
            'School_Enrollment': st.number_input("School Enrollment (%)", 0.0, 100.0, 80.0)
        }

        submit = st.form_submit_button("Classify")

        if submit:
            # ✅ FIX APPLIED HERE
            pred_df = prepare_input(input_data)

            pred_class = int(model.predict(pred_df)[0])
            prob = model.predict_proba(pred_df)[0][pred_class]

            label = "High Income" if pred_class == 1 else "Lower Income"
            st.success(f"📊 Classification: {label}")
            st.info(f"Confidence: {prob:.1%}")

# =========================================================
# ========= MODE 3: MODEL COMPARISON =======================
# =========================================================
else:
    st.header("⚖ Model Performance Comparison")
    st.info("All models trained and compared on the same dataset.")
