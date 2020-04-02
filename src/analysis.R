library(dplyr)
library(forecast)
library(ggfortify)
library(ggplot2)
library(quantmod)
options('getSymbols.warning4.0' = F)


getSymbols(Symbols = '^GSPC',
           src     = 'yahoo',
           auto.assign = T,
           from    = '2018-01-01',
           to      = '2020-01-01')
data <- GSPC[, 'GSPC.Close']
str(data)
head(data)
tail(data)
sp500 <- ts(data)


## Time Series Plot

sp500 %>% autoplot +
  xlab(label = 'Time') +
  ylab(label = 'Closing Price (US$)') +
  ggtitle(label = 'S&P 500 Time Series Plot')


## ACF Plot

sp500 %>% ggAcf +
  ggtitle(label = 'S&P 500 ACF Plot')


## PACF Plot

sp500 %>% ggPacf +
  ggtitle(label = 'S&P 500 PACF Plot')


### Portmanteau Test

sp500 %>% Box.test(lag = 25, type = 'Box-Pierce')


## Take first differences to remove non-stationary component

sp500_diff <- sp500 %>% diff(lag = 1)


## First Differenced Time Series Plot

sp500_diff %>% autoplot +
  xlab(label = 'Time') +
  ylab(label = 'Closing Price (US$)') +
  ggtitle(label = 'First Differenced S&P 500')


## First Difference ACF Plot

sp500_diff %>% ggAcf +
  ggtitle(label = 'First Differenced S&P 500 ACF Plot')


## First Difference PACF Plot

sp500_diff %>% ggPacf +
  ggtitle(label = 'First Differenced S&P 500 PACF Plot')


## Portmanteau Test

sp500_diff %>% Box.test(lag = 25, type = 'Box-Pierce')


# best_model <- readRDS('model/arima_model')
source('src/modeling.R')
saveRDS(best_model, file = 'model/arima_model')


## Residual Analysis

best_model %>% ggtsdiag


## Forecasting

### Retrieve the next 5 closing prices
getSymbols(Symbols = '^GSPC',
           src     = 'yahoo',
           auto.assign = T,
           from    = '2019-12-31',
           to      = '2020-01-09')
head(GSPC)
data <- ts(GSPC[, 'GSPC.Close'])
test <- data[2:6]
test_diff <- diff(data)[1:5]

### Evaluate MSE
forecast <- predict(best_model, n.ahead = 5)
forecast_pred <- forecast$pred
mse_diff <- mean((test_diff - forecast_pred)^2)
paste('First Differenced Mean Squared Error:', mse_diff)

last <- sp500[length(sp500)]
pred <- rep(Inf, 5)
for (x in 1:5) {
  pred[x] <- last + forecast_pred[x]
}
mse <- mean((test - pred)^2)

print(test)
print(pred)
