# Stock Price Prediction Using Machine Learning and Ensemble Methods

## Overview
This project focuses on predicting NTT stock prices using various machine learning models, including **ARIMA**, **Random Forest**, **LSTM**, and an **Ensemble Weighted Model**. The goal is to identify which models are most effective at predicting future stock prices and to create a robust approach that balances the strengths of different models.

## Project Objectives
- **Accurately predict NTT stock prices** using historical market data.
- Evaluate the effectiveness of multiple machine learning models for time series forecasting.
- Implement a final **Ensemble Model** to maximize accuracy and generalization.

## Models Implemented
1. **Baseline Naïve Model**: A benchmark model that predicts based on previous day's close.
2. **ARIMA**: Captures trends and seasonality of time series data.
3. **Random Forest**: Captures non-linear dependencies between features.
4. **LSTM**: A recurrent neural network for sequential data.
5. **Ensemble Weighted Model**: Combination of Random Forest and LSTM for improved predictions.

## Project Workflow
1. **Data Collection and Preprocessing**
   - **Data Source**: NTT stock market historical data.
   - **Preprocessing**: Included **scaling**, **handling missing values**, and **feature engineering**.
   - **Feature Engineering**: Added lag features, moving averages, technical indicators like **RSI**, **MACD**, and **Bollinger Bands**.

2. **Exploratory Data Analysis (EDA)**
   - Conducted **Volume Analysis**, detected **outliers**, and analyzed **trends and seasonality** using **seasonal decomposition**.
   - Visualized key insights through graphs and charts.

3. **Model Training**
   - **Random Forest**: Hyperparameter tuning using **GridSearchCV**.
   - **LSTM**: Utilized **dropout regularization**, **reduced learning rate**, and **early stopping** to improve generalization.
   - **Transformer-Based Model**: Implemented as an additional enhancement.

4. **Model Evaluation**
   - Evaluated models using **MAE**, **MSE**, **RMSE**, and **MAPE** metrics.
   - **Ensemble Model** outperformed all individual models with the lowest **MAE**.
   - Conducted **Residual Analysis** and **Error Distribution Analysis**.

5. **Improvement Measures**
   - Utilized **hyperparameter tuning** and **ensemble techniques** to reduce prediction errors.
   - The **Ensemble Model** achieved a **testing MAE of 0.8412**.

6. **Results and Future Work**
   - The **Ensemble Model** provided the most accurate predictions.
   - Future work includes experimenting with **Transformer-Based Models** for better long-term prediction and incorporating **macroeconomic indicators**.

## Repository Structure
- **data/**: Contains the historical stock data.
- **notebooks/**: Jupyter notebooks showing data exploration, preprocessing, and model training.
- **models/**: Saved model files for the trained models.
- **scripts/**: Python scripts for preprocessing, training, and evaluation.
- **README.md**: Project overview and instructions.

## Requirements
- Python 3.8+
- Required Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, Statsmodels, TensorFlow, Keras, PyTorch

Install the necessary libraries by running:
```bash
pip install -r requirements.txt
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/sauravraj188/stock-price-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd stock-price-prediction
   ```
3. Update the Path of stock_price.csv
4. Ensure that lstm22.keras and finalRF2.pkl file and stock_price_prediction_by_Saurav_Raj.ipynb have the same directory
5. Train models using the provided notebooks or scripts.


## Results
- **Random Forest Testing MAE**: 0.8536
- **LSTM Testing MAE**: 1.3206
- **Ensemble Weighted Testing MAE**: 0.8412
- **ARIMA Testing MAE**: 31.8825 (least effective)

The **Ensemble Model** had the best generalization, combining the strengths of **Random Forest** and **LSTM** to achieve the lowest error.

## Insights
- **ARIMA** struggled with handling high-frequency fluctuations, leading to higher error metrics.
- **Random Forest** and **LSTM** each contributed unique strengths to the final ensemble approach, improving overall robustness.
- **Ensemble Models** are generally advantageous for increasing predictive stability and reducing variance.

## Future Enhancements
- **Transformer-Based Time Series Models**: Further explore **Transformers** to capture long-term dependencies and complex temporal structures.
- **Feature Expansion**: Include additional external factors such as **macroeconomic indicators** or **sentiment analysis** data for better predictions.



## Contact
For questions or suggestions, feel free to reach out:
- **Author**: Saurav Raj
- **Email**: 21bcs188@iiitdmj.ac.in or official.sauravraj001@gmail.com

