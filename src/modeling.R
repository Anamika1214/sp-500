required_packages <- c('doParallel', 'foreach', 'parallel')
invisible(lapply(required_packages, library, character.only = T))
registerDoParallel(detectCores() - 1)

# LOOCV
folds = 1:(length(sp500_diff) - 1)
score <- foreach(k = folds, .packages = required_packages) %dopar% {
  # initialize data.frame for each thread
  spe <- data.frame(matrix(0, 7, 7), row.names = c('0', '1', '2', '3', '4', '5', '6'))
  colnames(spe) <- rownames(spe)
  train <- sp500_diff[1:k]
  val <- sp500_diff[(k + 1)]
  for (p in 0:6) {
    for (q in 0:(6 - p)) {
      model <- arima(x = train, order = c(p, 0, q))
      model_pred <- predict(model, n.ahead = 1)$pred
      spe[as.character(p), as.character(q)] <- (val - model_pred)^2
    }
  }
  return(spe)
}

mspe <- Reduce("+", score) / length(score)
print(mspe)

## Fit best model on the full train data
best_model <- arima(sp500_diff, order = c(2, 0, 2))
summary(best_model)
