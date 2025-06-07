# ❤️ Heart Disease Prediction using Machine Learning and Explainable AI

This project uses **Logistic Regression**, a classic machine learning algorithm, to predict the likelihood of heart disease based on patient health parameters. It includes feature-based predictions, explainability with **SHAP values**, and generates a complete PDF report including visuals, predictions, and recommendations.

---

## 🧠 Objective

The aim is to develop a lightweight, interpretable machine learning model that:
- Predicts the presence or absence of heart disease
- Explains the reasoning behind predictions using SHAP
- Calculates a health score
- Provides a personalized health suggestion
- Outputs a professional PDF report

---

## 📌 Features

- ✅ Logistic Regression-based prediction model  
- ✅ Takes patient input parameters and predicts risk  
- ✅ Health score based on risk probability  
- ✅ Personalized health advice based on prediction  
- ✅ SHAP visualizations for interpretability  
- ✅ Confusion Matrix and ROC Curve evaluation  
- ✅ Automatically generates a downloadable PDF report  

---

## 🛠️ Technologies Used

- **Python**
- **Pandas / NumPy** – Data handling
- **Scikit-learn** – Machine Learning (Logistic Regression, train-test split, evaluation metrics)
- **Matplotlib / Seaborn** – Visualization
- **SHAP** – Feature-level interpretability
- **ReportLab** – PDF report generation

---

## 🧾 Input Parameters (Features Used)

- Age  
- Sex  
- Chest Pain Type  
- Resting Blood Pressure  
- Cholesterol  
- Fasting Blood Sugar  
- Resting ECG Results  
- Maximum Heart Rate  
- Exercise-induced Angina  
- ST Depression  
- Slope of ST segment  
- Number of major vessels  
- Thalassemia  

These are collected via user input or test dataset, cleaned, and passed into the model for prediction.

---

## 📂 Dataset Used

The model is trained on the **Framingham Heart Study Dataset**, a well-known public dataset containing clinical data used to study heart disease risk factors.  
🔗 You can find the dataset on Kaggle or UCI repository.  
*Note: The dataset includes anonymized patient information and is used here for research and educational purposes only.*

---

## ⚙️ How the Model Works

1. **Data Preprocessing**:
   - Cleaned and normalized the Framingham Heart Study dataset
   - Handled missing values and encoded categorical features

2. **Model Training**:
   - Trained a **Logistic Regression model**
   - Evaluated using test data (train-test split)

3. **Prediction**:
   - User inputs health data manually or from a file
   - Model returns prediction and probability

4. **SHAP Explainability**:
   - Shows which features contributed most to the prediction
   - Generates a SHAP summary plot

5. **PDF Report**:
   - Final output includes patient summary, prediction result, SHAP plot, ROC curve, and health advice

---

## 📈 Model Evaluation

- **Accuracy**: Model’s overall performance  
- **Confusion Matrix**: Visual breakdown of true/false positives and negatives  
- **ROC Curve**: Plots True Positive Rate vs False Positive Rate  
- **AUC Score**: Performance metric ranging from 0 to 1  

---

## 📄 Output PDF Report Includes:

- Patient input details  
- Predicted disease status (Yes/No)  
- Risk probability score  
- Health score  
- Health suggestion  
- SHAP feature impact chart  
- Confusion matrix and ROC curve  

---

## 🚀 How to Run the Project

1. Clone the Repository:
```bash
git clone https://github.com/your-username/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
