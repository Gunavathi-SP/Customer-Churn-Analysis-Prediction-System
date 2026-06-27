# 📉 Customer Churn Prediction

A Machine Learning project that predicts whether a customer is likely to churn based on demographic, financial, and behavioral attributes. The project includes Exploratory Data Analysis (EDA), model training, evaluation, and an interactive Streamlit web application for real-time predictions.

---

## 🚀 Features

- 📊 Comprehensive Exploratory Data Analysis (EDA)
- 🤖 Multiple Machine Learning models trained and evaluated
- 📈 Model performance comparison using evaluation metrics
- 📉 ROC Curve and Confusion Matrix visualization
- ⭐ Feature Importance analysis
- 🌐 Interactive Streamlit web application
- 💾 Saved trained model for future predictions

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Plotly
- Streamlit

---

## 📂 Project Structure

```
Customer-Churn-Prediction/
│
├── churn_app.py                  # Streamlit web application
├── churn_eda.py                  # Exploratory Data Analysis
├── churn_ml.py                   # Model training and evaluation
├── churn_prediction.csv          # Dataset
│
├── models/
│   ├── best_model.pkl
│   └── meta.pkl
│
├── eda_plots/
│   ├── Churn Distribution
│   ├── Correlation Heatmap
│   ├── Boxplots
│   └── Other EDA visualizations
│
├── ml_plots/
│   ├── Model Comparison
│   ├── ROC Curves
│   ├── Confusion Matrix
│   ├── Feature Importance
│   └── Cross Validation
│
├── requirements.txt
└── README.md
```

---

## 📊 Machine Learning Workflow

1. Data Loading
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis
4. Feature Engineering
5. Model Training
6. Model Evaluation
7. Best Model Selection
8. Model Deployment using Streamlit

---

## 📈 Model Evaluation

The project evaluates multiple machine learning algorithms using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Cross Validation

The best-performing model is saved for deployment.

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
```

Move into the project directory

```bash
cd customer-churn-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run churn_app.py
```

---

## 📸 Results

The project includes visualizations such as:

- Customer Churn Distribution
- Numerical Feature Analysis
- Correlation Heatmap
- ROC Curve
- Confusion Matrix
- Feature Importance
- Model Comparison




his project useful, consider giving it a star!
