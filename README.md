# Decision-Tree-Algorithm-for-Time-Series-Forecasting-
This repository contains a project focused on **Time Series Forecast Analysis** using DT, that well known machine learning technique. The main objective of this project is to analyze time series data and develop forecasting model to predict future values of industrial response variable based on its historical trends and few of some external variables.

## Objectives
- Explore and identify the important data pre-processing steps when using ML models for time series forecasting.  
- Develop and implement **machine learning models** to forecast future values.  
- Compare the performance of machine learning models against **classical ARIMA models**.  
- Select the best-fit model that maximizes **forecast accuracy** using statistical metrics.

## Data Source 
The dataset is private industrial data containing: single response variable and four predictor variables.
The data is used to train the decision tree model and evaluate forecast accuracy against ARIMA benchmarks.

## Features
- Data cleaning and preprocessing of date-time data.    
- Implementation of a **Decision Tree–based forecasting model**:
  - Fit a full decision tree to the date-time data.  
  - Perform **feature selection** based on variable importance.  
  - **Hyperparameter tuning** by adjusting the complexity parameter (**cp**) to improve model accuracy.  
  - Evaluate the final model performance and compare it with classical ARIMA as a benchmark.
- Model comparison based on performance metrics such as MAE, RMSE, and MAPE.  
- Visualization of actual vs. predicted values.  

## Technologies Used
- **R**
- **Tidyverse / ggplot2** 
- **Forecast / Prophet / caret / keras** 
- **Time Series and Machine Learning libraries**

## Results and Key Insights
- **Data Preprocessing is Critical:**  
  Since machine learning models like decision trees do not inherently capture the temporal order of time series data, careful preprocessing is essential. Transformations, lag features, and proper handling of date-time information significantly impact model performance.  

- **Feature Selection Matters:**  
  Selecting the most important predictors based on variable importance improves model accuracy. In time series data, irrelevant or redundant features can lead to poor forecasts.  

- **Limitations of Decision Trees for Time Series:**  
  - Decision trees struggle to capture sequential dependencies and temporal trends.  
  - They may fail to identify peaks and troughs accurately in the response variable.  

- **Forecast Performance:**  
  Despite these limitations, the tuned decision tree model with carefully engineered features provides a significant improvement over naive approaches and serves as a strong baseline when compared to classical ARIMA models.  

- **Model Insights:**  
  Visualizations of predicted vs. actual values highlight how preprocessing and feature selection help the model better capture trends, even if extreme peaks are sometimes missed.

