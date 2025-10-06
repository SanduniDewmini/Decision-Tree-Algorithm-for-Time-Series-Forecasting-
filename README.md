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
