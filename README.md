# Retail Demand Forecasting with Festival and Social Media Trend Integration

## Problem Statement

Retail demand forecasting is a critical task for inventory management, supply chain optimization, and sales planning. Traditional forecasting approaches primarily rely on historical sales data and often fail to capture the influence of external factors such as festivals and social media trends.

This project aims to improve forecasting accuracy by integrating festival effects and simulated social media indicators into the prediction process.

---

## Objectives

* Forecast future retail demand using machine learning.
* Analyze the impact of festivals on sales performance.
* Simulate social media trend indicators and study their relationship with demand.
* Compare multiple machine learning models.
* Explain model predictions using SHAP.
* Develop an interactive Streamlit dashboard.

---

## Dataset

Primary Dataset: Walmart Store Sales Dataset

Features include:

* Store
* Date
* Weekly Sales
* Holiday Flag
* Temperature
* Fuel Price
* CPI
* Unemployment

Additional engineered features:

* Festival Features
* Social Media Trend Features
* Lag Features
* Rolling Statistics
* Interaction Features

---

## Methodology

### Phase 1: Data Understanding

* Data Loading
* Missing Value Analysis
* Duplicate Detection
* Outlier Detection
* Exploratory Data Analysis

### Phase 2: Festival Feature Engineering

Created:

* Festival Flag
* Festival Name
* Festival Score
* Days To Festival
* Festival Impact Score

Festivals Included:

* Diwali
* Dussehra
* Holi
* Ganesh Chaturthi
* Raksha Bandhan
* Janmashtami
* Christmas
* Black Friday
* Thanksgiving
* New Year
* Easter
* Halloween
* Labor Day
* Independence Day

### Phase 3: Social Media Trend Simulation

Generated:

* Trend Score
* Mention Count
* Engagement Rate
* Sentiment Score
* Influencer Promotion Flag
* Viral Campaign Flag

### Phase 4: Advanced Feature Engineering

Created:

* Year
* Month
* Week
* Quarter
* Day
* DayOfWeek
* WeekendFlag

Lag Features:

* Lag_1
* Lag_4
* Lag_12

Rolling Features:

* Rolling Mean
* Rolling Standard Deviation
* Moving Average

Interaction Features:

* Festival × Trend
* Festival × Sentiment
* Trend × Engagement

### Phase 5: Model Development

Models Trained:

1. Random Forest Regressor
2. XGBoost Regressor
3. LightGBM Regressor

### Phase 6: Evaluation

Metrics Used:

* MAE
* RMSE
* MAPE
* R² Score

### Phase 7: Explainable AI

Used SHAP to explain:

* Festival Impact
* Trend Impact
* Sentiment Impact
* Feature Importance

### Phase 8: Forecasting

Forecast Horizons:

* 7 Weeks
* 30 Weeks
* 90 Weeks

### Phase 9: Streamlit Dashboard

Features:

* Dataset Overview
* Sales Trend Analysis
* Festival Analytics
* Social Media Analytics
* Demand Forecasting
* SHAP Explainability
* CSV Download

---

## Model Performance

| Model         | MAE      | RMSE     | MAPE  | R²      |
| ------------- | -------- | -------- | ----- | ------- |
| Random Forest | 13207.55 | 24101.29 | 1.78% | 0.99616 |
| XGBoost       | 13147.24 | 22407.85 | 1.77% | 0.99668 |
| LightGBM      | 13236.23 | 23601.23 | 1.80% | 0.99632 |

Best Model: XGBoost Regressor

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* LightGBM
* SHAP
* Streamlit

---

## Future Scope

* Real-time social media integration using APIs.
* Real-time festival and event calendars.
* Deep learning forecasting models.
* Cloud deployment.
* Multi-store inventory optimization.

---

## Conclusion

The project successfully demonstrates that integrating festival information and social media trend indicators can improve retail demand forecasting. The XGBoost model achieved excellent predictive performance and the Streamlit dashboard provides an interactive decision-support system for demand planning.
