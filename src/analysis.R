## ----setup, message = FALSE, warning = FALSE---------------------------------------------------------------------------------------------------------------------------------
library(doSNOW)
library(forecast)
library(ggfortify)
library(gtools)
library(parallel)
library(quantmod)
library(tcltk)
options('getSymbols.warning4.0' = F)


## ----Load data---------------------------------------------------------------------------------------------------------------------------------------------------------------
getSymbols(Symbols = '^GSPC',
           src     = 'yahoo',
           auto.assign = T,
           from    = '2019-01-01',
           to      = '2020-01-01')
data <- GSPC[, 'GSPC.Close']
head(data)
tail(data)
sp500 <- ts(data)


## ----time series plot--------------------------------------------------------------------------------------------------------------------------------------------------------
ggtsdisplay(sp500, main = 'S&P 500 Index Closing Price')


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Box.test(sp500, lag = 12, type = 'Box-Pierce')


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
sp500 %>%
  diff() %>%
  ggtsdisplay(main = 'First Differenced S&P 500 Index Closing Price')


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
sp500 %>%
  diff() %>%
  Box.test(lag = 12, type = 'Box-Pierce')


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
tslm_linear <- tslm(sp500 ~ trend)
summary(tslm_linear)
fcast_linear <- forecast(tslm_linear, h = 5)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
autoplot(sp500) +
  autolayer(fitted(tslm_linear), series = 'Linear') +
  autolayer(fcast_linear, series = 'Linear') +
  xlab('Time') +
  ylab('Closing Stock Price') +
  ggtitle('S&P 500 Index Forecast') +
  guides(colour = guide_legend(title = ' '))


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
t <- time(sp500)
t1 <- ts(pmax(0, t - 37))
t2 <- ts(pmax(0, t - 205))
tslm_spline <- tslm(sp500 ~ t + I(t^2) + I(t^3) + I(t1^3) + I(t2^3))
summary(tslm_spline)

fcast_spline <- forecast(tslm_spline,
                         newdata = data.frame(t = t[length(t)] + seq(5),
                                              t1 = t1[length(t1)] + seq(5),
                                              t2 = t2[length(t2)] + seq(5)))



## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
autoplot(sp500) +
  autolayer(fitted(tslm_spline), series = 'Cubic Spline') +
  autolayer(fcast_spline, series = 'Cubic Spline') +
  xlab('Time') +
  ylab('Closing Stock Price') +
  ggtitle('S&P 500 Index Forecast') +
  guides(colour = guide_legend(title = ' '))


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ff_linear <- function(x, h) {
  forecast(tslm(x ~ trend), h = h)
}
e_linear <- tsCV(sp500, ff_linear, h = 5)
rmse_linear <- sqrt(mean(e_linear^2, na.rm = T))


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
fcast_ses <- ses(sp500, h = 5, level = 95)
summary(fcast_ses)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
autoplot(sp500) +
  autolayer(fitted(fcast_ses), series = 'SES') +
  autolayer(fcast_ses, series = 'SES') +
  xlab('Time') +
  ylab('Closing Stock Price') +
  ggtitle('S&P 500 Index Forecast') +
  guides(colour = guide_legend(title = ' '))


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
holt_fcast <- holt(sp500, h = 5, level = 95)
summary(holt_fcast)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
autoplot(sp500) +
  autolayer(fitted(holt_fcast), series = "Holt's") +
  autolayer(holt_fcast, series = "Holt's") +
  xlab('Time') +
  ylab('Closing Stock Price') +
  ggtitle('S&P 500 Index Forecast') +
  guides(colour = guide_legend(title = ' '))

## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
holt_fcast_damped <- holt(sp500, h = 5, level = 95, damped = T)
summary(holt_fcast_damped)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
autoplot(sp500) +
  autolayer(fitted(holt_fcast_damped), series = "Damped Holt's") +
  autolayer(holt_fcast_damped, series = "Damped Holt's") +
  xlab('Time') +
  ylab('Closing Stock Price') +
  ggtitle('S&P 500 Index Forecast') +
  guides(colour = guide_legend(title = ' '))


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
tsCV_ses <- tsCV(sp500, ses, h = 5)
rmse_ses <- sqrt(mean(tsCV_ses^2, na.rm = T))

tsCV_holt <- tsCV(sp500, holt, h = 5)
rmse_holts <- sqrt(mean(tsCV_holt^2, na.rm = T))

tsCV_holt_damped <- tsCV(sp500, holt, damped = T, h = 5)
rmse_damped_holts <- sqrt(mean(tsCV_holt_damped^2, na.rm = T))


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
cl <- makeSOCKcluster(detectCores() - 1)
registerDoSNOW(cl)

order <- permutations(5, 2, 0:4, repeats = T)
order <- lapply(1:nrow(order), function(i) order[i, ]) # Convert matrix to list
pb <- tkProgressBar(max = length(order))
opts <- list(progress = function(n) setTkProgressBar(pb, n))

score <- foreach(pq = order, .options.snow = opts, .packages = c('forecast')) %dopar% {
  p <- pq[1]
  q <- pq[2]
  aic <- Arima(sp500, order = c(p, 1, q))$aic
  farima <- function(x, h) {
    forecast(Arima(x, order = c(p, 1, q)), h = h)
  }
  e <- tsCV(sp500, farima, h = 5)
  rmspe <- sqrt(mean(e^2, na.rm = T))
  return(c(p, q, aic, rmspe))
}

close(pb)
stopCluster(cl)

result <- data.frame(Reduce(rbind, score), row.names = NULL)
colnames(result) <- c('p', 'q', 'AIC', 'RMSE')
print(result)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
head(result[order(result$RMSE),])


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
head(result[order(result$AIC),])


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
arima_model <- Arima(sp500, order = c(1, 1, 0), include.constant = T)
summary(arima_model)
rmse_arima <- min(result$RMSE)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
fcast_arima <- forecast(arima_model, h = 5, level = 95)
autoplot(sp500) +
  autolayer(fitted(fcast_arima), series = 'ARIMA(1,1,0)') +
  autolayer(fcast_arima, series = 'ARIMA(1,1,0)') +
  xlab('Time') +
  ylab('Closing Stock Price') +
  ggtitle('S&P 500 Index Forecast') +
  guides(colour = guide_legend(title = ' '))


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ggtsdiag(arima_model)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
checkresiduals(arima_model, lag = 12)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
cl <- makeSOCKcluster(detectCores() - 1)
registerDoSNOW(cl)

order <- 1:4
pb <- tkProgressBar(max = length(order))
opts <- list(progress = function(n) setTkProgressBar(pb, n))

score <- foreach(p = order, .options.snow = opts, .packages = c('forecast')) %dopar% {
  fnnetar <- function(x, h) {
    set.seed(2020)
    forecast(nnetar(y = x, p = p, size = 1), h = h)
  }
  e <- tsCV(sp500, fnnetar, h = 5)
  rmspe <- sqrt(mean(e^2, na.rm = T))
  return(c(p, rmspe))
}

close(pb)
stopCluster(cl)

result <- data.frame(Reduce(rbind, score), row.names = NULL)
colnames(result) <- c('p', 'RMSE')
print(result)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
set.seed(2020)
nnar_model <- nnetar(y = sp500, p = 1, size = 1, repeats = 200)
print(nnar_model)
rmse_nnar <- min(result$RMSE)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
fcast_nnar <- forecast(nnar_model, h = 5)
autoplot(sp500) +
  autolayer(fitted(fcast_nnar), series = 'NNAR(1,1)') +
  autolayer(fcast_nnar, series = 'NNAR(1,1)') +
  xlab('Time') +
  ylab('Closing Stock Price') +
  ggtitle('S&P 500 Index Forecast') +
  guides(colour = guide_legend(title = ' '))


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
rmse_cv <- setNames(data.frame(rmse_linear, rmse_ses, rmse_holts, rmse_damped_holts, rmse_arima, rmse_nnar, row.names = 'RMSE'),
                    c('Linear Regression', 'SES', "Holt's", "Damped Holt's", "ARIMA", 'NNAR'))
print(rmse_cv)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
getSymbols(Symbols = '^GSPC',
           src     = 'yahoo',
           auto.assign = T,
           from    = '2020-01-01',
           to      = '2020-03-31')
data <- GSPC[1:5, 'GSPC.Close']
head(data)
test <- as.vector(data)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
rmse_test <- data.frame(rbind(accuracy(fcast_ses$mean, test),
                              accuracy(fcast_arima$mean, test),
                              accuracy(fcast_nnar$mean, test),
                              accuracy(fcast_spline$mean, test)),
                        row.names = c('SES', 'ARIMA', 'NNAR', 'Cubic Spline'))
print(rmse_test[order(rmse_test$RMSE), ])


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
cbind(data.frame(fcast_arima), Actual = test)


## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ggplot(data.frame(data), aes(x = index(data))) +
  geom_line(aes(y = GSPC.Close, color = 'Actual')) +
  geom_line(aes(y = fcast_arima$mean, color = 'ARIMA')) +
  geom_line(aes(y = fcast_nnar$mean, color = 'NNAR')) +
  geom_line(aes(y = fcast_ses$mean, color = 'SES')) +
  xlab('Time') +
  ylab('Closing Stock Price') +
  ggtitle('S&P 500 Index Forecast vs Actual')

