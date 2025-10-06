# =======================
# Required Libraries
# =======================
library(readxl)        # For reading Excel files
library(tidyverse)     # Collection of packages for data manipulation and visualization
library(forecast)      # Time series forecasting tools
library(MLmetrics)     # For machine learning evaluation metrics
library(neuralnet)     # Neural network modeling
library(imputeTS)      # Handling missing values in time series
library(astsa)         # Applied statistical time series analysis
library(dplyr)         # Data manipulation
library(zoo)           # Working with regular and irregular time series
library(rpart)         # Recursive partitioning for classification and regression trees
library(rpart.plot)    # Plotting rpart trees
library(Metrics)       # Common ML evaluation metrics like RMSE, MAE, etc.
library(ggplot2)       # Advanced data visualization

# ===========================
# Data Import & Preprocessing
# ===========================
dataoriginal <- data.frame(read_excel("C:/Users/Administrator/Downloads/XXX-20250722T121102Z-1-001-20250807T032535Z-1-001/XXX-20250722T121102Z-1-001/XXX/DatasetApr25.xlsx"))

# Multivariate ts: columns 2:6 assumed c(X1,X2,X3,X4,Y)
data <- ts(dataoriginal[,2:6], frequency = 12, start = c(2005, 1))

# Interpolate missing values
data <- ts(na_interpolation(data), frequency = 12, start = c(2005, 1))

# ===========================
# STL Decomposition (Fixed)
# ===========================
# Custom function to remove trend using STL decomposition
stl_detrend <- function(series) {
  stl_decomp <- stl(series, s.window = "periodic", robust = TRUE) # Decompose series using STL
  list(
    detrended = stl_decomp$time.series[, "remainder"],   # remainder = series - trend - seasonal
    trend     = stl_decomp$time.series[, "trend"],
    seasonal  = stl_decomp$time.series[, "seasonal"]
  )
}

X1_result <- stl_detrend(data[, "X1"])
X2_result <- stl_detrend(data[, "X2"])
X3_result <- stl_detrend(data[, "X3"])
X4_result <- stl_detrend(data[, "X4"])
Y_result  <- stl_detrend(data[, "Y"])

# Detrended dataset (use remainders)
data_detrended <- cbind(
  X1 = X1_result$detrended,
  X2 = X2_result$detrended,
  X3 = X3_result$detrended,
  X4 = X4_result$detrended,
  Y  = Y_result$detrended
)
data_detrended <- ts(data_detrended, frequency = 12, start = c(2005, 1))

# Save components for Y
Y_trend    <- as.numeric(Y_result$trend)
Y_seasonal <- as.numeric(Y_result$seasonal)

# Sanity check: reconstruction should equal original (tiny numerical noise allowed)
Y_recon_check <- as.numeric(Y_result$detrended + Y_result$trend + Y_result$seasonal)
cat("Max abs diff (reconstruct vs original Y):",
    max(abs(Y_recon_check - as.numeric(data[,"Y"]))), "\n")

# ===========================
# Important Feature Selection (CCF)
# ===========================
# Cross correlation function to identify the important predictors when forecasting Y.
ccf(data_detrended[, "X1"], data_detrended[, "Y"], lag.max = 12, main = "CCF between x1 and y")
# From left side, some bars are above the blue lines.
# That means, Past values of x1 are correlated with future values of y.

ccf(data_detrended[, "X2"], data_detrended[, "Y"], lag.max = 12, main = "CCF between x2 and y")
# From left side, some bars are above the blue lines.
# This means: future values of x2 are correlated with current y.

ccf(data_detrended[, "X3"], data_detrended[, "Y"], lag.max = 12, main = "CCF between x3 and y")
# No significant correlation at negative lags. x3 does not help forecast y.

ccf(data_detrended[, "X4"], data_detrended[, "Y"], lag.max = 12, main = "CCF between x4 and y")
# Both sides shows significant spikes over all lags. That means X4 leads Y and Y influenced on X4. 
# X4 is importnt when forecasting Y. 

# X1, X2 and X4 are the variables that important when forecasting Y. 
# Now we want to identify how many lagged features of these variables we should add to our model. 

# ===========================
# Feature Engineering (with alignment kept)
# ===========================
dataNew_df <- as.data.frame(data_detrended)
dataNew_df$time_num <- as.numeric(time(data_detrended))       # e.g., 2005.000, 2005.083...
dataNew_df$ym       <- as.yearmon(time(data_detrended))        # yearmon for plotting/tables
dataNew_df$orig_idx <- seq_len(nrow(dataNew_df))               # <-- KEY: original row index

# Calendar features (no external packages)
m <- as.integer(format(dataNew_df$ym, "%m"))
y <- as.integer(format(dataNew_df$ym, "%Y"))
q <- (m - 1) %/% 3 + 1

dataFeat <- dataNew_df %>%
  mutate(
    month   = factor(m),
    quarter = factor(q),
    year    = factor(y),
    year_num = y
  )

# Lags to create
vars <- c("X1", "X2", "X4")
max_lag <- 12

for (v in vars) {
  for (l in 1:max_lag) {
    dataFeat <- dataFeat %>%
      mutate(!!paste0(v, "_lag_", l) := dplyr::lag(.data[[v]], l))
  }
}
for (l in 1:12) {
  dataFeat <- dataFeat %>%
    mutate(!!paste0("Y_lag_", l) := dplyr::lag(dataNew_df$Y, l))
}

# Drop rows made NA by lagging
data_model <- na.omit(dataFeat)

# ===========================
# Train-Test Split
# ===========================
n <- nrow(data_model)
train_size <- floor(0.8 * n)
train_data <- data_model[1:train_size, ]
test_data  <- data_model[(train_size + 1):n, ]

# ===========================
# Full Model (After Detrending)
# ===========================
predictors_full <- names(data_model)[grepl("X1_lag_|X2_lag_|X4_lag_|Y_lag_", names(data_model))]

# (Optionally) include calendar features in exploration
full_formula <- as.formula(paste("Y ~", paste(c(predictors_full, "month", "quarter", "year"), collapse = " + ")))
# Modeling formula (lags only)
full_formula  <- as.formula(paste("Y ~", paste(predictors_full, collapse = " + ")))

full_tree_model <- rpart(full_formula, data = train_data, method = "anova")
rpart.plot(full_tree_model)

# Importance table
importance_table <- data.frame(
  Variable   = names(full_tree_model$variable.importance),
  Importance = round(full_tree_model$variable.importance, 2),
  row.names = NULL
) %>% arrange(desc(Importance))
print(importance_table)

# ===========================
# Select Important Lags + Refine
# ===========================
important_vars <- names(full_tree_model$variable.importance)
selected_vars  <- intersect(predictors_full, important_vars)

# Use selected lags + numeric year to avoid factor level issues
refined_formula <- as.formula(paste("Y ~", paste(c(selected_vars, "year_num"), collapse = " + ")))

refined_tree_model <- rpart(
  refined_formula,
  data = train_data,
  method = "anova",
  control = rpart.control(cp = 0.001, minsplit = 10, maxdepth = 10)
)

# ===========================
# Prune & Predict
# ===========================
# Select the cp value that corresponds to the lowest cross-validation error (xerror)
# The cptable stores model metrics at different tree sizes:
#   - CP: complexity parameter tested
#   - nsplit: number of internal splits
#   - rel error: relative error on training data
#   - xerror: cross-validated error
#   - xstd: standard deviation of xerror
#
best_cp <- refined_tree_model$cptable[which.min(refined_tree_model$cptable[,"xerror"]), "CP"]
pruned_tree <- prune(refined_tree_model, cp = best_cp)

preds_det_test  <- predict(pruned_tree, newdata = test_data)   # predictions on the DETRENDED scale (remainder)

# ===========================
# Restore Trend + Seasonality (Aligned)
# ===========================
# Use the saved original row indices to align components EXACTLY
idx_test  <- test_data$orig_idx
idx_train <- train_data$orig_idx

# Test set reconstruction
preds_test_final   <- preds_det_test + Y_trend[idx_test] + Y_seasonal[idx_test]
actual_test_final  <- as.numeric(data[idx_test, "Y"])   # direct from original series (gold standard)

# Train set predictions and reconstruction
train_preds_det    <- predict(pruned_tree, newdata = train_data)
preds_train_final  <- train_preds_det + Y_trend[idx_train] + Y_seasonal[idx_train]
actual_train_final <- as.numeric(data[idx_train, "Y"])

# ===========================
# Evaluate (Test)
# ===========================
cat("MAPE (%):", mape(actual_test_final, preds_test_final) * 100, "\n")
cat("RMSE    :", rmse(actual_test_final, preds_test_final), "\n")
cat("MAE     :", mae(actual_test_final, preds_test_final), "\n")

# ===========================
# Final Forecast Table (Test)
# ===========================
forecast_table <- data.frame(
  Date     = test_data$ym,
  Actual   = round(actual_test_final, 2),
  Forecast = round(preds_test_final, 2)
)
forecast_table <- forecast_table %>%
  mutate(Error = round(Actual - Forecast, 2),
         Pct_Error = round((abs(Error) / pmax(1e-12, Actual)) * 100, 2))
head(forecast_table)

# ===========================
# Plot Actual vs Forecast (Test)
# ===========================
plot_data <- forecast_table %>%
  pivot_longer(cols = c("Actual", "Forecast"), names_to = "Type", values_to = "Value")

ggplot(plot_data, aes(x = Date, y = Value, color = Type)) +
  geom_line(size = 1.2) +
  labs(title = "Actual vs Forecast (Trend + Seasonality Restored, Aligned)",
       x = "Date", y = "Value") +
  theme_minimal() +
  theme(legend.position = "top",
        plot.title = element_text(hjust = 0.5, size = 14))

# ===========================
# Train Set Table & Plot
# ===========================
train_forecast_table <- data.frame(
  Date   = train_data$ym,
  Actual = round(actual_train_final, 2),
  Fitted = round(preds_train_final, 2)
) %>%
  mutate(Error = round(Actual - Fitted, 2),
         Pct_Error = round((abs(Error) / pmax(1e-12, Actual)) * 100, 2))
head(train_forecast_table)

train_plot_data <- train_forecast_table %>%
  pivot_longer(cols = c("Actual", "Fitted"), names_to = "Type", values_to = "Value")

ggplot(train_plot_data, aes(x = Date, y = Value, color = Type)) +
  geom_line(size = 1.2) +
  labs(title = "Actual vs Fitted (Trend + Seasonality Restored, Aligned)",
       x = "Date", y = "Value") +
  theme_minimal() +
  theme(legend.position = "top",
        plot.title = element_text(hjust = 0.5, size = 14))

