# 🌍 Machine Learning–Based Analysis and Prediction of Economic Development Using World Bank Data

## 📌 Project Overview

This project focuses on applying **machine learning techniques** to analyze and predict **economic development** using globally recognized data from the **World Bank World Development Indicators (WDI)** dataset.

The project aims to:

* Predict **GDP per capita** using key development indicators
* Classify countries into **income categories**
* Compare traditional and advanced machine learning models
* Evaluate model performance using standard metrics

The analysis demonstrates how machine learning can be used for **data-driven economic insights and decision support**.

---

## 📊 Dataset Information

* **Dataset Name:** World Development Indicators (WDI)
* **Dataset Provider:** World Bank Group
* **Dataset Platform:** World Bank Data360
* **Dataset Link:**
  [https://data360.worldbank.org/en/dataset/WB_WDI](https://data360.worldbank.org/en/dataset/WB_WDI)

### 🔹 Data Acquisition Method

The dataset is fetched **programmatically** using the **World Bank API** through the `wbgapi` Python library. The project retrieves indicator-wise data for multiple countries over the **last 20 years**, ensuring authenticity and reproducibility.

In case of API unavailability, the project supports **manual CSV upload** as a fallback option.

---

## 📌 Indicators Used

The following indicators were selected due to their relevance to economic development:

* GDP per capita (current US$) *(Target Variable)*
* Life expectancy at birth (years)
* Secondary school enrollment (%)
* Inflation, consumer prices (annual %)
* Energy use (kg of oil equivalent per capita)
* CO₂ emissions (metric tons per capita)

---

## 🧠 Machine Learning Models Implemented

### 🔹 Regression Models

* Linear Regression
* Polynomial Regression
* Random Forest Regressor
* Gradient Boosting Regressor
* Neural Network (MLP Regressor)

### 🔹 Classification Models

* Logistic Regression
* Random Forest Classifier
* Gradient Boosting Classifier
* Support Vector Machine (SVM)
* Neural Network (MLP Classifier)

---

## 📈 Model Evaluation Metrics

### 🔹 Regression

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

### 🔹 Classification

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## 🚀 Project Deployment

The project has been deployed to demonstrate real-time prediction and classification.

* **Deployment Platform:** Streamlit
* **Live Project Link:**
  👉 *(Add your deployed project link here)*

---

## 🗂️ Repository Structure

```
├── data/                  # Dataset files (if exported)
├── notebooks/             # Jupyter / Colab notebooks
├── src/                   # Machine learning scripts
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

*(Structure may vary depending on implementation)*

---

## 🛠️ Technologies Used

* **Programming Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn
* **Visualization:** Matplotlib, Plotly
* **Deployment:** Streamlit
* **Environment:** Google Colab / Jupyter Notebook

---

## 🔁 Reproducibility

* Dataset is publicly available from the World Bank
* Code is fully documented and reproducible
* Models can be retrained using updated data via the API

---

## 👩‍🎓 Author

**Tanishka Soni**

* Registration No: 12319510
* Program: B.Tech Computer Science Engineering
* Course: INT234 – Predictive Analysis
* Institution: Lovely Professional University

---

## 📜 License

This project is intended for **academic and educational purposes only**.
