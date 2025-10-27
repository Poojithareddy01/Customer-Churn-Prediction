# Customer-Churn-Prediction
Machine learning project that predicts customer churn using Python, Pandas, Scikit-learn, and XGBoost. Includes data preprocessing, feature engineering, and deployment as a Streamlit web app for interactive churn prediction and SHAP-based model explainability.
# 🧠 Customer Churn Prediction (Streamlit + Machine Learning)

This project predicts **customer churn** using machine learning techniques. It uses **Python, Pandas, Scikit-learn, and XGBoost** to build and evaluate a predictive model, and includes a **Streamlit web app** that allows users to upload customer data, generate churn predictions, and visualize key feature influences using SHAP values.

---

## 🚀 Project Overview
- Built an **end-to-end machine learning pipeline** to identify customers likely to churn.  
- Cleaned and preprocessed the **Telco Customer Churn dataset**.  
- Applied **SMOTE** for handling class imbalance.  
- Trained and tuned an **XGBoost model** for prediction.  
- Created an interactive **Streamlit app** for real-time churn prediction.  

---

## ⚙️ Tech Stack
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Streamlit, SHAP, Imbalanced-learn, Matplotlib, Seaborn  
- **Tools:** Jupyter Notebook, Anaconda, Git, GitHub  
- **Deployment:** Streamlit Web App  

---


git clone https://github.com/<your-username>/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
pip install -r requirements.txt
jupyter notebook Customer_Churn_Prediction.ipynb
streamlit run churn_app.py
