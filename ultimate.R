# 1
library(dplyr)
library(tidyverse)
view(longley)

# 2
plot(x = longley$Employed, y = longley$GNP.deflator, col = 'blue')
plot(x = longley$Employed, y = longley$GNP, col = 'blue')
plot(x = longley$Employed, y = longley$Unemployed, col = 'blue') # no
plot(x = longley$Employed, y = longley$Armed.Forces, col = 'blue') # no
plot(x = longley$Employed, y = longley$Population, col = 'blue')
plot(x = longley$Employed, y = longley$Year, col = 'blue')

# variable most correlated with Employed is GNP.deflator, GNP, Population

# 3
err <- function(val, pr) {
  fit.stats <- as.data.frame(cbind(val, pr)) %>%
    rename(Actuals = val, Predicted.Values = pr) %>%
    mutate(error = Actuals - Predicted.Values,  # y - yhat
           squared.error = error^2)

  SSE <- sum(fit.stats$squared.error)

  RMSE <- sqrt(mean(fit.stats$squared.error))

  return(RMSE)
}


modgd <- lm(Employed ~ GNP.deflator, data = longley)
err(longley$Employed, modgd$fitted.values) # 0.814

modg <- lm(Employed ~ GNP, data = longley)
err(longley$Employed, modg$fitted.values) # 0.614 BEST MODEL (LESS ERROR)

modp <- lm(Employed ~ Population, data = longley)
err(longley$Employed, modp$fitted.values) # 0.947

# Best model is GNP
# 4
y <- as.matrix(longley$Employed) ## response
X <- as.matrix(cbind(rep(1,length(longley$Employed)), longley[,c('GNP')]))

# 5
beta <- solve(t(X) %*% X) %*% t(X) %*% y
beta
# Beta is
#     [,1]
# [1,] 51.84358978
# [2,]  0.03475229
summary(modg)
# Estimated from summary is identical
#             Estimate Std. Error t value Pr(>|t|)    
# (Intercept) 51.843590   0.681372   76.09  < 2e-16 ***
#   GNP          0.034752   0.001706   20.37 8.36e-12 ***
longley2 <- longley %>%
  mutate(predEmp = beta[1] + beta[2] * GNP)
err(longley$Employed, longley2$predEmp) # 0.614 same as from lm function
