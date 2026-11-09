# 🔥 Wildfire Size Prediction at Ignition - New Zealand

**Applying Machine Learning to identify potentially large wildfires using historical environmental, weather, geographical and temporal data.**

## 🌲 About the Project

This project investigates whether machine learning can classify the potential size of wildfires in New Zealand using information available at or around the time of fire detection.

Using the New Zealand Wildfire Occurrence Database (WOD), the project focuses on distinguishing between:

- 🟢 **Small Fires:** Final burned area below 1 hectare.
- 🔴 **Large Fires:** Final burned area of 1 hectare or more.

A key challenge is the significant class imbalance, with large wildfires representing only a small proportion of historical records.

## 🚀 What I Worked On

- **Exploratory Data Analysis:** Investigated wildfire distributions, geographical patterns, burned-area trends, missing values and changes in feature availability over time.
- **Feature Engineering:** Implemented cyclical encoding for month and hour, One-Hot Encoding for categorical variables and Standard Scaling for numerical features.
- **Data Preprocessing:** Built Scikit-learn pipelines using median and most-frequent-value imputation.
- **Feature Selection:** Used Random Forest feature importance to identify relevant predictors.
- **Model Development:** Implemented and compared six classification algorithms:
  - Logistic Regression
  - Random Forest
  - Extra Trees
  - Gradient Boosting
  - Hist Gradient Boosting
  - XGBoost
- **Class Imbalance Handling:** Experimented with class weighting and SMOTE to improve the detection of large wildfires.
- **Hyperparameter Tuning:** Applied RandomizedSearchCV with 3-fold cross-validation to explore Gradient Boosting configurations.
- **Model Evaluation:** Used PR-AUC, ROC-AUC, Precision, Recall, F1-score and confusion matrices to assess model performance.

## 📊 Model Performance

The Gradient Boosting classifier achieved the best validation PR-AUC among the six initial models.

| Evaluation Metric | Test Score |
|---|---:|
| Accuracy | 96.97% |
| Precision (Large Fires) | 67.58% |
| Recall (Large Fires) | 49.95% |
| F1-Score | 57.44% |
| ROC-AUC | 0.8929 |
| PR-AUC | 0.5220 |

<img width="550" height="400" alt="image" src="https://github.com/user-attachments/assets/4d3e866f-98be-4a0b-a699-6d18bc4a76c1" />


**Key Finding:** Despite achieving high overall accuracy, identifying rare large wildfires remained challenging. The model detected approximately half of the actual large fires, highlighting the difficulty of learning from highly imbalanced environmental datasets.

## 🗓️ Chronological Data Validation

To evaluate how well the models generalise to future wildfire records, the dataset was split chronologically rather than randomly.

| Dataset | Years |
|---|---|
| Training | 1998-2017 |
| Validation | 2018-2019 |
| Testing | 2020-2025 |

## 🛠️ Technologies & Libraries

`Python` `Jupyter Notebook` `Pandas` `NumPy` `Scikit-learn` `XGBoost` `Imbalanced-learn` `Matplotlib` `Seaborn`

## 🎯 Key Learning Outcomes

- Working with real-world environmental datasets containing missing values and highly imbalanced classes.
- Building end-to-end machine learning preprocessing and classification pipelines.
- Comparing ensemble learning algorithms and evaluating their performance.
- Understanding the importance of temporal validation and minority-class evaluation.
- Investigating the limitations of machine learning when predicting rare events.
