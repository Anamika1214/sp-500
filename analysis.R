## ----setup, message = FALSE, warning = FALSE-------------------------------------------------------------------------------------------------------
library(doSNOW)
library(forecast)
library(ggfortify)
library(parallel)
library(quantmod)
library(tcltk)
options('getSymbols.warning4.0' = F)


## ----Load data-------------------------------------------------------------------------------------------------------------------------------------
getSymbols(Symbols = '^GSPC',
           src     = 'yahoo',
           auto.assign = T,
           from    = '2019-01-01',
           to      = '2020-01-01')
data <- GSPC[, 'GSPC.Close']
head(data)
tail(data)
sp500 <- ts(data)


## ----time series plot------------------------------------------------------------------------------------------------------------------------------
ggtsdisplay(sp500, main = 'S&P 500 Index Closing Price')


## --------------------------------------------------------------------------------------------------------------------------------------------------
Box.test(sp500, lag = 25, type = 'Box-Pierce')


## --------------------------------------------------------------------------------------------------------------------------------------------------
sp500 %>%
  diff() %>%
  ggtsdisplay(main = 'First Differenced S&P 500 Index Closing Price')


## --------------------------------------------------------------------------------------------------------------------------------------------------
sp500 %>%
  diff() %>%
  Box.test(lag = 25, type = 'Box-Pierce')


## --------------------------------------------------------------------------------------------------------------------------------------------------
cluster <- makeSOCKcluster(detectCores(logical = T) - 1)
registerDoSNOW(cluster)

nfolds <- length(sp500) - 1
# Leave-one-out
# kfolds <- 21:nfolds
# 12-Fold (~1 month rolling window)
kfolds <- round(seq(5, nfolds, length.out = 52))

# For Progress Bar window:
pb <- tkProgressBar(max = length(kfolds))
opts <- list(progress = function(n) setTkProgressBar(pb, n))
# For console output:
# opts <- list(progress = function(n) cat(sprintf('Fold %d is complete\n', n)))

fit_arima <- function(x, p, q) {
  model <- tryCatch({
      return(Arima(x, order = c(p, 1, q), include.constant = T))
  }, error = function(e) {
    tryCatch({
      return(Arima(x, order = c(p, 1, q), include.constant = T, method = 'ML'))
    }, error = function(e) {
      return(Arima(x, order = c(p, 1, q), include.constant = T, method = 'ML', transform.pars = F))
    })
  })
  return(model)
}

score <- foreach(k = kfolds, .options.snow = opts, .packages = c('forecast')) %dopar% {
  # initialize data.frame for each thread
  spe <- data.frame(matrix(0, 5, 5), row.names = c('0', '1', '2', '3', '4'))
  colnames(spe) <- rownames(spe)
  aic <- data.frame(matrix(0, 5, 5), row.names = c('0', '1', '2', '3', '4'))
  colnames(aic) <- rownames(aic)
  # Split sp500 data into train and validation set
  train <- sp500[1:k]
  validation <- sp500[k + 1]
  for (p in 0:4) {
    for (q in 0:4) {
      if (p == 0 && q == 0) next # Skip ARIMA(0, 1, 0)
      model <- fit_arima(x = train, p, q)
      aic[as.character(p), as.character(q)] <- model$aic
      y_hat <- forecast(model, h = 1)$mean[1]
      spe[as.character(p), as.character(q)] <- (y_hat - validation)^2
    }
  }
  return(list(spe, aic))
}
close(pb)

rmspe_list <- sapply(score, function(x) x[1])
rmspe <- sqrt(Reduce('+', rmspe_list) / length(score))

aic_list <- sapply(score, function(x) x[2])
aic <- Reduce('+', aic_list) / length(score)

result <- data.frame(matrix(ncol = 4, nrow = 0))
colnames(result) <- c('p', 'q', 'RMSPE', 'AIC')
for (i in 1:5) {
  for (j in 1:5) {
    if (i == 1 && j == 1) next
    result[nrow(result) + 1,] <- c(i - 1, j - 1, rmspe[i, j], aic[i, j])
  }
}


## --------------------------------------------------------------------------------------------------------------------------------------------------
print(result[order(result$RMSPE)[1:5], ])


## --------------------------------------------------------------------------------------------------------------------------------------------------
print(result[order(result$AIC)[1:5], ])


## --------------------------------------------------------------------------------------------------------------------------------------------------
best_model <- Arima(sp500, order = c(1, 1, 0), include.constant = T)
summary(best_model)


## --------------------------------------------------------------------------------------------------------------------------------------------------
ggtsdiag(best_model)


## --------------------------------------------------------------------------------------------------------------------------------------------------
checkresiduals(best_model, lag = 25)


## --------------------------------------------------------------------------------------------------------------------------------------------------
getSymbols(Symbols = '^GSPC',
           src     = 'yahoo',
           auto.assign = T,
           from    = '2020-01-01',
           to      = '2020-03-31')
data <- GSPC[1:5, 'GSPC.Close']
head(data)
test <- as.vector(data)


## --------------------------------------------------------------------------------------------------------------------------------------------------
forecast <- forecast(best_model, h = 5, level = 95)
autoplot(forecast) +
  ggtitle(label = 'S&P 500 Index Closing Price Forecast') +
  ylab(label = 'Price') +
  xlab(label = 'Time')


## --------------------------------------------------------------------------------------------------------------------------------------------------
pred <- as.vector(forecast$mean)
accuracy(pred, test)


## --------------------------------------------------------------------------------------------------------------------------------------------------
forecast_data <- data.frame(date = index(data),
                            price = c(pred, test),
                            predicted = pred,
                            actual = test)
ggplot(forecast_data, aes(x = date, y = predicted)) + 
  geom_line(aes(color = 'Predicted')) +
  geom_line(aes(y = actual, color = 'Actual')) +
  xlab(label = 'Time') +
  ylab(label = 'Closing Price (US$)') +
  ggtitle(label = 'S&P 500 Index Closing Price Forecast') +
  theme(legend.title = element_blank())

