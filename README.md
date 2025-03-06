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

```bash
/prehospital-trauma-ai
├── model.py          # AI model implementation (XGBoost + LightGBM + Random Forest)
├── data_loader.py    # Data loading and preprocessing (KTDB & external datasets)
├── train.py          # Model training script
├── evaluate.py       # Performance evaluation and metric computation
├── shap_analysis.py  # SHAP analysis for model interpretability
├── README.md         # Project overview and usage guide
