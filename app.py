import streamlit as st
import pandas as pd
import numpy as np
import wbgapi as wb
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage

# Machine Learning Imports
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, 
                             accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)

# --- CONFIGURATION ---
st.set_page_config(page_title="WDI Advanced Analytics", layout="wide", page_icon="🌍")

# --- CUSTOM CSS FOR "GOOD UI" ---
st.markdown("""
<style>
    .main-header {font-size:36px; font-weight:bold; color:#2c3e50;}
    .sub-header {font-size:24px; font-weight:bold; color:#34495e;}
    .metric-box {background-color:#f0f2f6; padding:15px; border-radius:10px; border-left: 5px solid #4CAF50;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🌍 World Development Indicators: Syllabus Suite</div>', unsafe_allow_html=True)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("📚 Syllabus Units")
unit_selection = st.sidebar.radio("Go to Unit:", [
    "Unit I: Data Prep",
    "Unit II: Regression (Supervised)",
    "Unit III: Classification (Supervised)",
    "Unit IV: Clustering (Unsupervised)",
    "Unit V: PCA & Neural Networks",
    "Unit VI: Model Performance"
])

# --- GLOBAL DATA LOADING ---
@st.cache_data
def load_wdi_data():
    indicators = {
        'NY.GDP.PCAP.CD': 'GDP_Per_Capita',
        'SP.DYN.LE00.IN': 'Life_Expectancy',
        'EN.ATM.CO2E.PC': 'CO2_Emissions',
        'FP.CPI.TOTL.ZG': 'Inflation',
        'EG.USE.PCAP.KG.OE': 'Energy_Use',
        'SE.SEC.ENRR': 'School_Enrollment'
    }
    
    # Fetch data
    df = wb.data.DataFrame(indicators.keys(), mrv=10, labels=False, db=2)
    
    # Reshape
    df.reset_index(inplace=True)
    melted = df.melt(id_vars=['economy', 'series'], var_name='year', value_name='value')
    pivoted = melted.pivot_table(index=['economy', 'year'], columns='series', values='value')
    pivoted = pivoted.rename(columns=indicators)
    final_df = pivoted.reset_index()
    
    # Initial cleaning (drop rows where Target is missing)
    final_df = final_df.dropna(subset=['GDP_Per_Capita'])
    return final_df

try:
    raw_df = load_wdi_data()
    # Fill features for stability (Unit I logic)
    df = raw_df.fillna(raw_df.mean(numeric_only=True))
    
    # Define X and y
    feature_cols = ['Life_Expectancy', 'CO2_Emissions', 'Inflation', 'Energy_Use', 'School_Enrollment']
    # Ensure columns exist
    valid_features = [c for c in feature_cols if c in df.columns]
    X = df[valid_features]
    y = df['GDP_Per_Capita']
    
except Exception as e:
    st.error(f"Critical Data Error: {e}")
    st.stop()

# --- UNIT I: DATA PREPARATION ---
if unit_selection == "Unit I: Data Prep":
    st.header("🛠️ Unit I: Introduction & Data Preparation")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw Data Preview")
        st.dataframe(raw_df.head(10))
    with col2:
        st.subheader("Preprocessing Controls")
        imputer = st.selectbox("Missing Value Handling", ["Mean Imputation", "Drop Rows", "Median Imputation"])
        scaler_type = st.radio("Feature Scaling", ["None", "StandardScaler (Z-Score)", "MinMaxScaler (0-1)"])
        
        # Dynamic Scaling Visualization
        data_to_plot = X.copy()
        if scaler_type == "StandardScaler (Z-Score)":
            scaler = StandardScaler()
            data_to_plot = pd.DataFrame(scaler.fit_transform(X), columns=valid_features)
            st.success("Applied Standard Scaling (Mean=0, Std=1)")
        elif scaler_type == "MinMaxScaler (0-1)":
            scaler = MinMaxScaler()
            data_to_plot = pd.DataFrame(scaler.fit_transform(X), columns=valid_features)
            st.success("Applied Min-Max Scaling (Range 0 to 1)")
            
    st.subheader("Feature Distributions (Post-Processing)")
    st.bar_chart(data_to_plot.head(20))

# --- UNIT II: REGRESSION ---
elif unit_selection == "Unit II: Regression (Supervised)":
    st.header("📈 Unit II: Regression Analysis")
    
    # Model Selection
    reg_type = st.selectbox("Select Regression Model", ["Simple Linear", "Multiple Linear", "Polynomial Regression"])
    
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    
    if reg_type == "Polynomial Regression":
        degree = st.slider("Polynomial Degree", 2, 5, 2)
        poly = PolynomialFeatures(degree=degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        model.fit(X_train_poly, y_train)
        preds = model.predict(X_test_poly)
    else:
        # Simple vs Multiple is just about how many features we pass. 
        # Here we pass all valid_features, so it is effectively Multiple.
        # For 'Simple', we could restrict to 1 feature.
        if reg_type == "Simple Linear":
            feat = st.selectbox("Select Single Feature", valid_features)
            X_train = X_train[[feat]]
            X_test = X_test[[feat]]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    
    # Display Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{mae:.0f}")
    col2.metric("MSE", f"{mse:.0f}")
    col3.metric("RMSE", f"{rmse:.0f}")
    col4.metric("R² Score", f"{r2:.4f}")
    
    # Visuals
    fig, ax = plt.subplots()
    ax.scatter(y_test, preds, alpha=0.6, color='blue')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel("Actual GDP")
    ax.set_ylabel("Predicted GDP")
    ax.set_title(f"Actual vs Predicted ({reg_type})")
    st.pyplot(fig)

# --- UNIT III: CLASSIFICATION ---
elif unit_selection == "Unit III: Classification (Supervised)":
    st.header("🏷️ Unit III: Classification Algorithms")
    
    # Target Engineering
    threshold = st.slider("High Income Threshold ($)", 
                          int(y.min()), int(y.max()), int(y.median()))
    y_class = (y > threshold).astype(int)
    
    st.info(f"Target: 0 = Low Income, 1 = High Income (Above ${threshold})")
    
    clf_name = st.selectbox("Select Classifier", 
                            ["K-Nearest Neighbors (KNN)", "Naive Bayes", "Decision Tree", "Support Vector Machine (SVM)"])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42)
    
    if "KNN" in clf_name:
        k = st.slider("Select K", 1, 20, 5)
        model = KNeighborsClassifier(n_neighbors=k)
    elif "Naive Bayes" in clf_name:
        model = GaussianNB()
    elif "Decision Tree" in clf_name:
        model = DecisionTreeClassifier()
    elif "SVM" in clf_name:
        model = SVC()
        
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    # Metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.2%}")
    m2.metric("Precision", f"{prec:.2%}")
    m3.metric("Recall", f"{rec:.2%}")
    m4.metric("F1 Score", f"{f1:.2%}")
    
    # Confusion Matrix Plot
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, preds)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

# --- UNIT IV: CLUSTERING ---
elif unit_selection == "Unit IV: Clustering (Unsupervised)":
    st.header("🧩 Unit IV: Clustering & Pattern Detection")
    
    clust_type = st.selectbox("Algorithm", ["K-Means Clustering", "Hierarchical Clustering", "Association Rules (Demo)"])
    
    if clust_type == "K-Means Clustering":
        # Elbow Method
        st.subheader("Elbow Method (Finding Optimal K)")
        wcss = []
        for i in range(1, 11):
            km = KMeans(n_clusters=i, init='k-means++', random_state=42)
            km.fit(X)
            wcss.append(km.inertia_)
            
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(1, 11), wcss, marker='o')
        ax.set_title('The Elbow Method')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('WCSS')
        st.pyplot(fig)
        
        # Actual Clustering
        k = st.slider("Select Final K", 2, 10, 3)
        km = KMeans(n_clusters=k)
        clusters = km.fit_predict(X)
        
        # 2D Visualization
        st.subheader("Cluster Visualization")
        fig, ax = plt.subplots()
        plt.scatter(X['Life_Expectancy'], y, c=clusters, cmap='viridis')
        plt.xlabel("Life Expectancy")
        plt.ylabel("GDP")
        st.pyplot(fig)
        
    elif clust_type == "Hierarchical Clustering":
        st.subheader("Dendrogram (Agglomerative)")
        # Limit to 50 samples for readability
        X_small = X.head(50)
        linked = linkage(X_small, 'ward')
        fig = plt.figure(figsize=(10, 5))
        dendrogram(linked)
        st.pyplot(fig)
        
    elif clust_type == "Association Rules (Demo)":
        st.warning("Note: WDI Data is continuous. Showing Market Basket Simulation for Syllabus compliance.")
        transactions = [['Milk', 'Bread'], ['Milk', 'Bread', 'Diapers'], ['Milk', 'Diapers'], ['Bread', 'Diapers']]
        st.write("Synthetic Transactions:", transactions)
        st.write("Pattern: {Milk} -> {Bread} (High Confidence)")

# --- UNIT V: PCA & NEURAL NETWORKS ---
elif unit_selection == "Unit V: PCA & Neural Networks":
    st.header("🧠 Unit V: Dim. Reduction & Neural Networks")
    
    tab1, tab2 = st.tabs(["PCA (Dim Reduction)", "Neural Network (MLP)"])
    
    with tab1:
        st.subheader("Principal Component Analysis (PCA)")
        pca = PCA(n_components=2)
        components = pca.fit_transform(X)
        
        fig, ax = plt.subplots()
        scatter = ax.scatter(components[:, 0], components[:, 1], c=y, cmap='plasma')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        plt.colorbar(scatter, label='GDP')
        st.pyplot(fig)
        st.write(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
        
    with tab2:
        st.subheader("Multi-Layer Perceptron (Feedforward NN)")
        hidden_layers = st.slider("Hidden Layer Size", 10, 100, 50)
        iter = st.slider("Max Iterations", 200, 1000, 500)
        
        # Scale Data (Crucial for NN)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
        
        mlp = MLPRegressor(hidden_layer_sizes=(hidden_layers,), max_iter=iter, random_state=42)
        
        if st.button("Train Neural Network"):
            with st.spinner("Training Neural Net..."):
                mlp.fit(X_train, y_train)
                preds = mlp.predict(X_test)
                score = r2_score(y_test, preds)
                st.success(f"Neural Network Trained! R² Score: {score:.4f}")

# --- UNIT VI: MODEL PERFORMANCE ---
elif unit_selection == "Unit VI: Model Performance":
    st.header("🏆 Unit VI: Advanced Performance Evaluation")
    
    st.subheader("Cross-Validation (Bias-Variance Check)")
    k_folds = st.slider("Number of Folds (K)", 2, 10, 5)
    
    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=k_folds, scoring='r2')
    
    st.write("Cross Validation Scores:", scores)
    st.metric("Average Accuracy (Bias)", f"{scores.mean():.4f}")
    st.metric("Standard Deviation (Variance)", f"{scores.std():.4f}")
    
    st.divider()
    
    st.subheader("Ensemble Method: Random Forest")
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X, y)
    
    # Feature Importance Plot
    importance = pd.Series(rf.feature_importances_, index=valid_features)
    st.bar_chart(importance)
    st.caption("Which features matter most for GDP?")