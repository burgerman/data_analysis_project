# FXUSDCAD Exchange Rate Forecasting using a Hybrid Prophet-LSTM Model

## Project Overview

This project aims to forecast the daily FXUSDCAD (U.S. Dollar to Canadian Dollar) exchange rate. It employs a hybrid forecasting model that combines Facebook's Prophet library and a Long Short-Term Memory (LSTM) neural network.

The core idea is to leverage the strengths of both models:
1.  **Prophet:** To model the main trend, seasonalities (weekly, monthly, quarterly), and the impact of external economic indicators (regressors).
2.  **LSTM:** To learn and predict the remaining non-linear patterns and noise from the residuals (errors) of the Prophet model.

The final forecast is the sum of the prediction from the Prophet model and the LSTM model's prediction on the residuals, creating a more accurate and robust hybrid forecast.

## Methodology

The forecasting process is divided into several key steps:

### 1. Data Preparation and Exploration
- The primary dataset (`group_project_training_dataset_v2.csv`) is loaded, containing daily data for various economic indicators.
- Initial preprocessing involves handling missing values and converting interest rate percentages into float values.
- Statistical tests (ANOVA and t-tests) are performed to analyze the significance of various regressors (e.g., interest rates, inflation) on the target variable, `FXUSDCAD`.

### 2. Prophet Model Forecasting
- A Prophet model is created and configured with custom seasonalities and holidays.
- Several external regressors are added to the model to improve its predictive power:
    - `canada_interest_rate`
    - `us_interest_rate`
    - `goods_inflation`
    - `services_inflation`
    - `NASDAQCOM`
- The model is fitted to the historical data, and a forecast is generated. The residuals (the difference between actual and predicted values) are calculated and stored.

### 3. LSTM Model for Residuals
- The residuals from the Prophet model are used as the target series for an LSTM network.
- A `Bidirectional LSTM` layer is used to capture patterns from the residual data in both forward and backward directions.
- The LSTM model is trained to predict the next day's residual based on a sequence of previous residuals.

### 4. Hybrid Forecast and Evaluation
- The predictions from the trained LSTM model (residual predictions) are added back to the initial Prophet forecast (`yhat`).
- This combination forms the final hybrid forecast.
- The performance of the Prophet, LSTM (on residuals), and the final Hybrid model are evaluated and compared using standard regression metrics:
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)

## File Structure
```
.
├── datasets/
│   ├── group_project_training_dataset_v2.csv  # Main dataset used for training
│   └── ... (other raw data files)
├── notebooks/
│   ├── group_project_notebook_v5.ipynb      # The main Jupyter Notebook with all the code
│   └── FXUSDCAD_model.tf/                     # Saved TensorFlow model from the notebook
└── README.md                                # This file
```

## How to Run

### Prerequisites
You need to have Python installed, preferably in a virtual environment.

### Dependencies
Install the required libraries using pip:
```bash
pip install pandas matplotlib prophet tensorflow scikit-learn
```

### Execution
1.  Ensure all the data files are present in the `datasets/` and `notebooks/` directories as per the file structure.
2.  Open and run the Jupyter Notebook `notebooks/group_project_notebook_v5.ipynb` from top to bottom.

The notebook will perform all the steps from data loading and analysis to model training, evaluation, and saving the final model.
