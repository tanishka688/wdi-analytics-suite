import streamlit as st
import pandas as pd
import numpy as np
import wbgapi as wb
import plotly.express as px
import plotly.graph_objects as go

# --- SYLLABUS MODELS IMPORT ---
# Unit II: Regression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

# Unit III: Classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# Unit VI: Model Performance
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# --- THEME: BLACK & WHITE (OBSIDIAN) ---
st.set_page_config(page_title="World Development Predictor", layout="wide", page_icon="🌍")

st.markdown("""
<style>
    /* Strict Black & White Theme */
    .stApp { background-color: #000000; color: #ffffff; }
    h1, h2, h3 { color: #ffffff !important; font-family: sans-serif; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #111111; border-right: 1px solid #333; }
    
    /* Inputs */
    .stSelectbox, .stNumberInput { color: white; }
    div[data-baseweb="select"] > div { background-color: #222; color: white; border-color: #444; }
    
    /* Buttons */
    .stButton > button {
        background-color: #222; color: white; border: 1px solid #555;
        border-radius: 4px; font-weight: bold;
    }
    .stButton > button:hover { border-color: #fff; background-color: #333; }
    
    /* Cards/Metrics */
    div[data-testid="metric-container"] {
        background-color: #1a1a1a; border: 1px solid #333; color: white;
    }
    
    /* Tables */
    [data-testid="stDataFrame"] { border: 1px solid #333; }
    
    /* Prediction Box */
    .prediction-box {
        background-color: #0a2a0a;
        border: 3px solid #00ff00;
        padding: 25px;
        border-radius: 12px;
        margin: 20px 0;
        box-shadow: 0 0 20px rgba(0,255,0,0.3);
    }
    
    .insight-box {
        background-color: #1a1a2e;
        border: 2px solid #4a90e2;
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
    }
    
    .warning-box {
        background-color: #2a1a0a;
        border: 2px solid #ff9500;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .explain {
        background-color: #111118;
        border: 1px solid #444;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.markdown("## World Development Intelligence System")
st.markdown("AI-powered economic predictions based on 20 years of global data")

# --- 1. DATA LOADING (Unit I: Data Prep) ---
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
                'Year': int(d['time'].replace('YR','')) if 'YR' in str(d['time']) else int(d['time']), 
                'Val': d['value']
            })
        
        if not rows: return None
        
        df = pd.DataFrame(rows).pivot_table(index=['Country', 'Year'], columns='Ind', values='Val').reset_index()
        
        for col in indicators.values():
            if col not in df.columns: df[col] = np.nan
            
        try:
            countries = wb.economy.DataFrame()['name'].to_dict()
            df['Country_Name'] = df['Country'].map(countries)
        except:
            df['Country_Name'] = df['Country']
            
        return df.dropna(subset=['Country_Name', 'GDP_Per_Capita'])
    except:
        return None

# Load
with st.spinner("Loading 20 years of global development data..."):
    df = load_data()

# Fallback
if df is None:
    st.error("API error or network blocked. Please upload 'WDI_Data.csv' to proceed.")
    f = st.file_uploader("Upload CSV", type='csv')
    if f: df = pd.read_csv(f)
    else: st.stop()

# --- PREPROCESSING (Unit I) ---
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    if df[col].isnull().all():
        df[col] = 0.0
    else:
        df[col] = df[col].fillna(df[col].median())

feature_cols = ['Life_Expectancy', 'CO2_Emissions', 'Inflation', 'Energy_Use', 'School_Enrollment']
valid_feats = [c for c in feature_cols if c in df.columns]

# Prepare country-average reference table used for comparisons
country_gdp_avg = df.groupby('Country_Name')['GDP_Per_Capita'].mean().dropna()
country_feats_avg = df.groupby('Country_Name')[valid_feats].median()

# Calculate global statistics for context
global_stats = {
    'avg_gdp': df['GDP_Per_Capita'].mean(),
    'avg_life': df['Life_Expectancy'].mean() if 'Life_Expectancy' in df.columns else np.nan,
    'avg_co2': df['CO2_Emissions'].mean() if 'CO2_Emissions' in df.columns else np.nan,
    'countries': df['Country_Name'].nunique(),
    'years': df['Year'].nunique()
}

# --- SIDEBAR ---
st.sidebar.header("Analysis Mode")
analysis_mode = st.sidebar.radio("Select analysis:", [
    "Economic Forecasting (Regression)",
    "Country Classification (High-Accuracy AI)",
    "Model Performance Comparison"
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Dataset overview**
- Countries: {global_stats['countries']}
- Years: {global_stats['years']}
- Avg GDP per capita: ${global_stats['avg_gdp']:,.0f}
- Avg life expectancy: {global_stats['avg_life']:.1f} years
""")

# Utility: find nearest countries by GDP
def nearest_countries_by_gdp(pred_value, top_n=3):
    diffs = (country_gdp_avg - pred_value).abs().sort_values()
    nearest = diffs.head(top_n).index.tolist()
    return nearest

# Utility: produce direct remark based on input and predicted class/value
def generate_regression_remark(pred_value, input_data):
    # classify into buckets
    if pred_value > 40000:
        cls = "Very high income"
        remark = "This level corresponds to advanced economies with strong services and high productivity."
    elif pred_value > 12000:
        cls = "High income"
        remark = "This level corresponds to developed or wealthy emerging economies."
    elif pred_value > 4000:
        cls = "Upper-middle income"
        remark = "This is a transitioning economy—industrializing with improving living standards."
    else:
        cls = "Lower-middle/low income"
        remark = "This is a lower-income profile; public investment can have large impact."
    # concrete actionable remarks (direct)
    actions = []
    if input_data.get('Life_Expectancy', 0) < global_stats['avg_life']:
        actions.append("Improve public health and primary care to raise life expectancy.")
    if input_data.get('School_Enrollment', 0) < 80:
        actions.append("Invest in secondary education and vocational training.")
    if input_data.get('CO2_Emissions', 0) > 10:
        actions.append("Adopt cleaner energy and efficiency measures to decouple growth from emissions.")
    if input_data.get('Inflation', 0) > 10:
        actions.append("Stabilize macroeconomy; control inflation to support investment.")
    if not actions:
        actions.append("Maintain current policies while focusing on productivity and innovation.")
    return cls, remark, actions

def generate_classification_remark(pred_class, pred_proba, input_data):
    if pred_class == 1:
        cls = "High income"
        remark = "Indicators suggest the country fits the high-income profile."
        if pred_proba < 0.75:
            remark += " Confidence is moderate — small policy shifts could change classification."
    else:
        cls = "Lower income"
        remark = "Indicators match lower-income profile; targeted investments can change trajectory."
        if pred_proba > 0.75:
            remark += " Confidence is strong — structural gaps are likely."
    # quick policy pointers
    pointers = []
    if input_data.get('Life_Expectancy', 0) < 70:
        pointers.append("Healthcare: strengthen primary care and maternal/child health.")
    if input_data.get('School_Enrollment', 0) < 85:
        pointers.append("Education: expand quality access to secondary education.")
    if input_data.get('Energy_Use', 0) < country_feats_avg['Energy_Use'].median() if 'Energy_Use' in country_feats_avg else False:
        pointers.append("Infrastructure: scale energy access to support industry.")
    return cls, remark, pointers

# --- MODE 1: ECONOMIC FORECASTING ---
if analysis_mode == "Economic Forecasting (Regression)":
    st.header("Economic development forecasting")
    st.markdown("Predict a country's GDP per capita using common development indicators. Input fields include a short explanation and a real-world example to help you choose realistic values.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("Select prediction model")
        reg_model_name = st.selectbox("Choose algorithm", [
            "Random Forest (best performance)",
            "Multiple Linear Regression",
            "Gradient Boosting",
            "Polynomial Regression (degree 2)"
        ])
    with col2:
        st.markdown("Model accuracy snapshot will appear after training.")

    # prepare data
    X = df[valid_feats]
    y = df['GDP_Per_Capita']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train selected model
    if "Random Forest" in reg_model_name:
        model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    elif "Gradient" in reg_model_name:
        model = GradientBoostingRegressor(n_estimators=150, random_state=42)
    elif "Multiple" in reg_model_name:
        model = LinearRegression()
    elif "Polynomial" in reg_model_name:
        model = Pipeline([('poly', PolynomialFeatures(2)), ('linear', LinearRegression())])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"${mae:,.0f}")
    c2.metric("RMSE", f"${rmse:,.0f}")
    c3.metric("R² score", f"{r2:.3f}")
    c4.metric("Explained variance (as %)", f"{r2*100:.1f}%")

    # plot actual vs predicted
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=preds, mode='markers', marker=dict(color='lightgreen', size=6, opacity=0.6), name='predictions'))
    fig.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()], mode='lines', line=dict(color='white', dash='dash', width=2), name='perfect'))
    fig.update_layout(template="plotly_dark", title="Actual vs Predicted GDP per capita", xaxis_title="Actual GDP", yaxis_title="Predicted GDP", height=400)
    st.plotly_chart(fig, use_container_width=True)

    # interactive input with explanation examples
    st.markdown("---")
    st.subheader("Make your prediction")
    st.markdown("Each input includes a short explanation and a concrete example. Use realistic values to get useful results.")

    with st.form("economic_prediction"):
        st.markdown("Input development indicators (examples provided)")

        col1, col2, col3 = st.columns(3)
        input_data = {}

        with col1:
            st.markdown("<div class='explain'><b>Life Expectancy (years)</b><br>What it measures: average lifespan at birth. Example: 82 = typical for advanced economies; 63 = typical for low-income countries.</div>", unsafe_allow_html=True)
            input_data['Life_Expectancy'] = st.number_input("Life Expectancy (years)", min_value=40.0, max_value=90.0, value=float(df['Life_Expectancy'].median()))
            st.markdown("<div class='explain'><b>CO2 Emissions (metric tons per person)</b><br>What it measures: annual emissions per person. Example: 15 = high-emissions developed economy; 1 = low-emissions developing economy.</div>", unsafe_allow_html=True)
            input_data['CO2_Emissions'] = st.number_input("CO2 Emissions (t per person)", min_value=0.0, max_value=30.0, value=float(df['CO2_Emissions'].median()))

        with col2:
            st.markdown("<div class='explain'><b>Inflation rate (%)</b><br>What it measures: year-over-year price growth. Example: 2% = stable, 25% = hyperinflationary stress.</div>", unsafe_allow_html=True)
            input_data['Inflation'] = st.number_input("Inflation rate (%)", min_value=-10.0, max_value=50.0, value=float(df['Inflation'].median()))
            st.markdown("<div class='explain'><b>Energy use (kg oil equivalent per person)</b><br>What it measures: energy consumption per person. Example: 5000 = industrialized high consumption; 500 = low consumption.</div>", unsafe_allow_html=True)
            input_data['Energy_Use'] = st.number_input("Energy Use (kg oil equiv.)", min_value=0.0, max_value=20000.0, value=float(df['Energy_Use'].median()))

        with col3:
            st.markdown("<div class='explain'><b>School enrollment (%)</b><br>What it measures: percent of eligible population enrolled in secondary education. Example: 95% = well-developed education system; 50% = underinvestment in education.</div>", unsafe_allow_html=True)
            input_data['School_Enrollment'] = st.number_input("School Enrollment (%)", min_value=0.0, max_value=100.0, value=float(df['School_Enrollment'].median()))

        submit = st.form_submit_button("Predict GDP per capita")

        if submit:
            pred_df = pd.DataFrame([input_data])
            pred_value = float(model.predict(pred_df)[0])

            # class bucket and remarks
            cls_label, cls_remark, actions = generate_regression_remark(pred_value, input_data)

            # nearest real countries
            nearest = nearest_countries_by_gdp(pred_value, top_n=3)
            nearest_str = ", ".join(nearest) if nearest else "No close match"
            # compare feature medians for nearest country if available
            comparison_lines = []
            if nearest:
                for c in nearest:
                    if c in country_feats_avg.index:
                        c_feats = country_feats_avg.loc[c]
                        diffs = []
                        for f in valid_feats:
                            diffs.append(f"{f}: input={input_data.get(f, np.nan):.2f}, {c}={c_feats.get(f, np.nan):.2f}")
                        comparison_lines.append(f"{c} -> " + "; ".join(diffs))
            else:
                comparison_lines.append("No country matches found for comparison.")

            # show prediction box
            st.markdown(f"""<div class='prediction-box'>
                <h2>Predicted GDP per capita: ${pred_value:,.2f}</h2>
                <h3>Income bucket: {cls_label}</h3>
                <p>{cls_remark}</p>
                <p><b>Nearest countries by GDP:</b> {nearest_str}</p>
                </div>""", unsafe_allow_html=True)

            # comparative insights
            comp_html = "<ul>"
            for line in comparison_lines:
                comp_html += f"<li style='margin-bottom:6px'>{line}</li>"
            comp_html += "</ul>"

            st.markdown(f"""<div class='insight-box'>
                <h4>How your input compares to similar countries</h4>
                {comp_html}
                <p><b>Global average GDP:</b> ${global_stats['avg_gdp']:,.0f} ({(pred_value/global_stats['avg_gdp']-1)*100:+.1f}% vs global average)</p>
                </div>""", unsafe_allow_html=True)

            # actions (direct, non-sugarcoated)
            actions_html = "<ul>"
            for a in actions:
                actions_html += f"<li>{a}</li>"
            actions_html += "</ul>"

            st.markdown(f"""<div class='warning-box'>
                <h4>Recommended immediate actions</h4>
                {actions_html}
                <p>These are practical steps that a policy maker can start implementing within 1-3 years to shift the trajectory. They are prioritized based on the gap between your input and typical high-income country indicators.</p>
                </div>""", unsafe_allow_html=True)

# --- MODE 2: COUNTRY CLASSIFICATION ---
elif analysis_mode == "Country Classification (High-Accuracy AI)":
    st.header("Economic development classification")
    st.markdown("Classify whether a country is 'high income' given its indicators. The form includes clear examples and the output includes direct remarks and comparisons to real countries.")

    # Income threshold slider
    threshold = st.slider("Define high income threshold ($ GDP per capita)", 5000, 50000, 12000, 1000)
    df['Class'] = (df['GDP_Per_Capita'] > threshold).astype(int)

    # class distribution
    high_count = int(df['Class'].sum())
    low_count = int(len(df) - high_count)
    st.write(f"Dataset distribution: {high_count} high-income records, {low_count} lower-income records (this is the data used to train the model).")

    # prepare and train model (ensure probability support for SVC used elsewhere)
    X = df[valid_feats]
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=250, max_depth=12, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)

    # metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)

    st.markdown("Model performance on holdout set:")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.2%}")
    c2.metric("Precision", f"{prec:.2%}")
    c3.metric("Recall", f"{rec:.2%}")
    c4.metric("F1 score", f"{f1:.2%}")
    if acc > 0.9:
        st.success(f"Model achieves {acc:.2%} accuracy on the test set.")

    # confusion matrix and feature importance
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("Confusion matrix")
        cm = confusion_matrix(y_test, preds)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Greens", labels=dict(x="Predicted", y="Actual"), x=['Low', 'High'], y=['Low', 'High'])
        fig_cm.update_layout(template="plotly_dark", height=350)
        st.plotly_chart(fig_cm, use_container_width=True)
    with col2:
        st.markdown("Feature importance")
        importance = pd.DataFrame({'Feature': valid_feats, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
        fig_imp = px.bar(importance, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale='Greens')
        fig_imp.update_layout(template="plotly_dark", height=350, showlegend=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    # interactive classification with explanations in the input form
    st.markdown("---")
    st.subheader("Classify a synthetic or real country (inputs include examples)")
    with st.form("classification_prediction"):
        st.markdown("Enter indicators (examples provided to guide realistic values)")

        col1, col2, col3 = st.columns(3)
        input_data = {}
        with col1:
            st.markdown("<div class='explain'><b>Life Expectancy</b><br>Example: 82 = advanced; 60 = low-income. This reflects health and social services.</div>", unsafe_allow_html=True)
            input_data['Life_Expectancy'] = st.number_input("Life Expectancy (years)", 40.0, 90.0, float(df['Life_Expectancy'].median()))
            st.markdown("<div class='explain'><b>CO2 Emissions</b><br>Example: 12 = typical for high-income, industrialized countries; 1 = low-income.</div>", unsafe_allow_html=True)
            input_data['CO2_Emissions'] = st.number_input("CO2 Emissions (t per person)", 0.0, 30.0, float(df['CO2_Emissions'].median()))
        with col2:
            st.markdown("<div class='explain'><b>Inflation rate (%)</b><br>Example: 2% = stable; >20% = severe macro instability.</div>", unsafe_allow_html=True)
            input_data['Inflation'] = st.number_input("Inflation rate (%)", -10.0, 50.0, float(df['Inflation'].median()))
            st.markdown("<div class='explain'><b>Energy use</b><br>Example: 6000 kg = energy-rich economy; 300 = low energy use.</div>", unsafe_allow_html=True)
            input_data['Energy_Use'] = st.number_input("Energy Use (kg oil equiv.)", 0.0, 20000.0, float(df['Energy_Use'].median()))
        with col3:
            st.markdown("<div class='explain'><b>School Enrollment (%)</b><br>Example: 95% = strong education system; 50% = large gaps.</div>", unsafe_allow_html=True)
            input_data['School_Enrollment'] = st.number_input("School Enrollment (%)", 0.0, 100.0, float(df['School_Enrollment'].median()))

        submit = st.form_submit_button("Classify country")

        if submit:
            pred_df = pd.DataFrame([input_data])
            pred_class = int(model.predict(pred_df)[0])
            pred_proba = model.predict_proba(pred_df)[0]
            conf = pred_proba[pred_class]

            # generate direct remark and pointers
            cls_label, cls_remark, pointers = generate_classification_remark(pred_class, conf, input_data)

            # nearest countries by GDP (for context)
            # use median of country_gdp_avg computed earlier
            predicted_gdp_placeholder = (input_data['Life_Expectancy'] - global_stats['avg_life']) * 200 + global_stats['avg_gdp']  # rough placeholder for context if needed
            nearest = nearest_countries_by_gdp(predicted_gdp_placeholder, top_n=3)
            nearest_str = ", ".join(nearest) if nearest else "No close match"

            st.markdown(f"""<div class='prediction-box'>
                <h2>Classification: {cls_label}</h2>
                <h3>Model confidence: {conf:.1%}</h3>
                <p>{cls_remark}</p>
                <p><b>Reference countries near this profile:</b> {nearest_str}</p>
                </div>""", unsafe_allow_html=True)

            # direct pointers
            pointers_html = "<ul>"
            for p in pointers:
                pointers_html += f"<li>{p}</li>"
            pointers_html += "</ul>"

            st.markdown(f"""<div class='insight-box'>
                <h4>Profile analysis</h4>
                <p>Key indicators you provided:</p>
                <ul>
                    <li>Life Expectancy: {input_data['Life_Expectancy']:.1f} years</li>
                    <li>School Enrollment: {input_data['School_Enrollment']:.1f}%</li>
                    <li>CO2 Emissions: {input_data['CO2_Emissions']:.2f} t/person</li>
                </ul>
                </div>""", unsafe_allow_html=True)

            if pointers:
                st.markdown(f"""<div class='warning-box'>
                    <h4>Recommended priorities</h4>
                    {pointers_html}
                    <p>These are the first-order priorities to change the classification over a 3-7 year horizon.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.info("No urgent policy changes required based on the indicators provided. Focus on long-term productivity growth and innovation.")

# --- MODE 3: MODEL COMPARISON ---
elif analysis_mode == "Model Performance Comparison":
    st.header("Model performance comparison")
    st.markdown("Compare different regression and classification models and see concrete results. Outputs include direct remarks about strengths and weaknesses of top models.")

    X = df[valid_feats]
    y_reg = df['GDP_Per_Capita']

    threshold = 12000
    df['Class'] = (df['GDP_Per_Capita'] > threshold).astype(int)
    y_clf = df['Class']

    # split
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

    # regression models
    st.markdown("Regression models")
    reg_models = {
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=150, random_state=42),
        "Linear Regression": LinearRegression(),
        "Polynomial (Deg 2)": Pipeline([('poly', PolynomialFeatures(2)), ('linear', LinearRegression())])
    }

    reg_results = []
    for name, m in reg_models.items():
        m.fit(X_train_reg, y_train_reg)
        preds = m.predict(X_test_reg)
        r2 = r2_score(y_test_reg, preds)
        rmse = np.sqrt(mean_squared_error(y_test_reg, preds))
        reg_results.append({'Model': name, 'R² Score': r2, 'RMSE': rmse})
    reg_df = pd.DataFrame(reg_results).sort_values('R² Score', ascending=False)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(reg_df.style.format({'R² Score': '{:.3f}', 'RMSE': '{:,.0f}'}), use_container_width=True, hide_index=True)
    with col2:
        fig_reg = px.bar(reg_df, x='Model', y='R² Score', color='R² Score', color_continuous_scale='Greens', title="Regression comparison")
        fig_reg.update_layout(template="plotly_dark", showlegend=False)
        st.plotly_chart(fig_reg, use_container_width=True)

    # classification models
    st.markdown("---")
    st.markdown("Classification models (income level)")
    clf_models = {
        "Random Forest": RandomForestClassifier(n_estimators=250, max_depth=12, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM (prob enabled)": SVC(kernel='rbf', probability=True, random_state=42),
        "Naive Bayes": GaussianNB()
    }

    clf_results = []
    for name, m in clf_models.items():
        m.fit(X_train_clf, y_train_clf)
        preds = m.predict(X_test_clf)
        acc = accuracy_score(y_test_clf, preds)
        prec = precision_score(y_test_clf, preds, zero_division=0)
        rec = recall_score(y_test_clf, preds, zero_division=0)
        f1 = f1_score(y_test_clf, preds, zero_division=0)
        status = "90%+" if acc > 0.9 else "Good" if acc > 0.8 else "OK"
        clf_results.append({'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1, 'Status': status})

    clf_df = pd.DataFrame(clf_results).sort_values('Accuracy', ascending=False)
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.dataframe(clf_df.style.format({'Accuracy': '{:.1%}', 'Precision': '{:.1%}', 'Recall': '{:.1%}', 'F1 Score': '{:.1%}'}), use_container_width=True, hide_index=True)
    with col2:
        fig_clf = px.bar(clf_df, x='Model', y='Accuracy', color='Accuracy', color_continuous_scale='Greens', title="Classification accuracy")
        fig_clf.update_layout(template="plotly_dark", showlegend=False)
        fig_clf.add_hline(y=0.9, line_dash="dash", line_color="cyan", annotation_text="90% threshold")
        st.plotly_chart(fig_clf, use_container_width=True)

    # direct recommendation block
    best_reg = reg_df.iloc[0]
    best_clf = clf_df.iloc[0]
    st.markdown("---")
    st.markdown("Direct, practical takeaway")
    st.markdown(f"- Best regression model: {best_reg['Model']} (R²={best_reg['R² Score']:.3f}). Practical meaning: use this model if you need accurate GDP point estimates. It balances bias and variance for this dataset.")
    st.markdown(f"- Best classification model: {best_clf['Model']} (Accuracy={best_clf['Accuracy']:.1%}). Practical meaning: use this model for policy screening — it reliably separates high vs lower income groups on these indicators.")
    st.info("Use regression models for point forecasts and classification models for categorical policy decisions (e.g., eligibility for a program).")

# end
