## Split data into train and validation set

# Using 2 folds (~250 rolling window)
train_18 <- sp500_diff[1:245]
val_18 <- sp500_diff[246:250]
train <- sp500_diff[1:497]
val <- sp500_diff[498:502]


## Time Series Cross Validation for Model Selection

ar_order <- vector()
ma_order <- vector()
aic <- vector()
mspe <- vector()
index <- 0
for (p in 0:6) {
  for (q in 0:(6 - p)) {
    index <- index + 1
    ar_order[index] <- p
    ma_order[index] <- q
    # 2018
    model_18 <- arima(x = train_18,
                      order = c(p, 0, q))
    model_18_pred <- predict(model_18,
                             n.ahead = 5)$pred
    mspe_18 <- mean((val_18 - model_18_pred)^2)
    # 2018-2019
    model_1819 <- arima(x = train,
                        order = c(p, 0, q))
    model_1819_pred <- predict(model_1819,
                               n.ahead = 5)$pred
    mspe_1819 <- mean((val - model_1819_pred)^2)
    # Save mean scores
    aic[index] <- mean(c(model_18$aic, model_1819$aic))
    mspe[index] <- mean(c(mspe_18, mspe_1819))
  }
}
order <- order(mspe)
result <- data.frame(p = ar_order[order],
                     q = ma_order[order],
                     AIC = aic[order],
                     MSPE = mspe[order])
print(distinct(result)[1:10, ])


## Fit best model on the full train data

best_model <- arima(sp500_diff, order = c(2, 0, 2))
summary(best_model)
