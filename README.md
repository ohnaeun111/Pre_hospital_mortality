# Real-time AI Model for Prehospital Trauma Mortality Predcition

This project presents a **real-time AI model for predicting emergency room (ER) mortality in trauma patients using only prehospital data**.  
Traditional triage tools rely on **limited physiological data and subjective assessments**, often leading to misclassification.  
To address these limitations, we developed an **interpretable AI model utilizing 21 prehospital variables**, validated across multiple institutions and countries.  

---

## Overview  
We implemented a **gradient boosting ensemble model (XGBoost + LightGBM + Random Forest)** to predict trauma mortality before hospital arrival.  
The model was designed for **real-time application**, achieving an **inference speed of 1.06 seconds**, making it feasible for use in emergency scenarios.  

Compared to the conventional **Shock Index (SI) triage tool**, our AI model demonstrated **superior predictive performance (AUROC: 0.9433)** and was externally validated across **multiple institutions and nations**.  

---

## Study Setting & Dataset  
- **Development & Internal Validation**: Trained on the **Korean Trauma Data Bank (KTDB, 227,567 patients)**  
- **External Validation**:  
  - **South Korea**: 4 regional trauma centers (8,867 patients)  
  - **Australia**: 1 Level 1 trauma center (3,786 patients)  
- **Inclusion Criteria**: Patients with available **prehospital trauma care data**  
- **Exclusion Criteria**: Non-trauma-related deaths and **patients transferred to another hospital**  

---

## Trauma Data Prepreocessing
- **This script anonymizes and processes trauma-related prehospital data into model-ready features.**  
- **Main Steps**:  
  - *Classification of time-of-injury as day, night, or missing*
  - *Age group binning (0–120 by 5-year intervals)*
  - *Selected categorical variables (e.g., gender, injury type, intentionality) are one-hot encoded.*
    - Example categories include:
      - gender_male
      - intentionality_{accident, suicide, assault, others, unknown, missing}
      - injury_type_{blunt, penetrating, burn, others, unknown, missing}
  - *Vital Signs Categorization*
     - Six core vital signs are discretized into clinically meaningful bins:
       - SBP (systolic_bp): 0, 1–49, 50–75, 76–89, 90+
       - DBP (diastolic_bp): 0, 1–29, 30–45, 46–59, 60+
       - Pulse (pulse): 0, 1–29, 30–59, 60–100, 101–119, 120+
       - Respiration Rate (respiration): 0, 1–5, 6–9, 10–29, 30+
       - Body Temperature (body_temp, °C): 0, 0–24, 24–28, 28–32, 32–35, 35–37, 38+
       - Oxygen Saturation (spo2, %): 0, 1–80, 81–90, 91–95, 96+
     - Each bin is encoded as a binary variable (e.g., sbp_50_75 = 1).
     - Additionally, two missing value flags are generated for each vital sign:
       - {vital}_uncheckable: if the value is -1
       - {vital}_unchecked: if the value is -9

---

## Features  

 **Dynamic Weight Adjustment**  
   - The ensemble model dynamically adjusts weight distribution, reducing model bias during training  

 **Real-time Prediction & Interpretability**  
   - **Fast inference speed of 1.06 seconds** for real-time emergency applications  
   - **SHAP (Shapley Additive Explanations) analysis** for improved model interpretability  

 **Multi-Institutional & International Validation**  
   - Model performance validated on **independent datasets from South Korea and Australia**  

 **Superior to Conventional Triage Tools**  
   - AI Model **AUROC: 0.9433, Sensitivity: 87.4%, Specificity: 86.3%**  
   - Shock Index (SI) **AUROC: 0.7117**  

 **Key Predictors Identified by SHAP Analysis**  
   - **Oxygen Saturation**  
   - **AVPU Scale (Alert, Verbal, Pain, Unresponsive)**  
   - **Transport Mode**  
   - **Injury Mechanism**  

---

## File Structure  
- **`model.py`** : Implements the AI model using XGBoost, LightGBM, and Random Forest.  
- **`data_loader.py`** : Handles data loading, filtering, and preprocessing for internal and external validation datasets.  
- **`train.py`** : Trains the AI model using ensemble learning and optimizes class weights.  
- **`evaluate.py`** : Evaluates the trained model's performance and generates key metrics.   
- **`README.md`** : Provides an overview of the project and usage instructions.
- **`Prehospital-AI model`** : Folder containing trained 5-fold model files. (.pkl)
- **`Sample_dataset.csv`** : Sample prehospital data for testing.
- **`Preprocess Trauma Data`** : Code to preprocess structured prehospital trauma data.

---

## Web demo
- 
