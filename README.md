# california-housing-price-prediction
This is my ML project on california housing price prediction

Motivation:Predicting housing price has always been a hot topic in data analytics field. During our undergraduate studies, we have witnessed a great change in housing prices in California. For instance, in San Diego where we spent 4 years studying, the housing prices have increased 50% from 2015 to 2019. Thus, we are pretty curious about if we can apply Machine Learning techniques on correctly forecasting sales price of houses. On the other hand, since many real estate businesses rely on deciding the appropriate prices to sell their houses, we believe our project can potentially help them in this regard. Moreover, investors who are interested in real estates and individual householder who wants to buy/sell house can benefit from our results as well.


Conclusion: The best model for our dataset is forward stepwise from data. The OSR^2 is 0.9998046, the RMSE is 0.005636024, and MAE is 0.00411799. Back to our dependent variable, the absolute average error is e^0.00411799 percentage. The following graph shows its prediction to dependent variable.
We used the bootstrap to assess the performance of the final model on test set.
The bias, std. error, and 95% of confidence interval for each metric is as follows:
RMSE: -7.817053e-05, 1.074243e-03, ( 0.0036, 0.0078 )
MAE: 1.841090e-05, 8.352736e-04, ( 0.0024, 0.0056 )
OSR^2: -9.891208e-07, 7.234604e-05, ( 0.0024, 0.0056 )



Guide:
Load the data into R and install packages as written in the code, set seed at 242, then the results can be reproduced.
