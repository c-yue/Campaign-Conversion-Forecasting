# Campaign-Conversion-Forecasting

#### Assignment 2 (of Marketing Analytics) is a predictive modeling exercise. 
#### My objective is to maximize the financial performance of a specific campaign for a charity.

My goal is to predict who is likely to make a donation to the charity for the fundraising campaign “C189” (%), 
and how much money they are likely to give if they do (€). 
By combining these two predictions (% * €), I will obtain an expected revenue from each individual.

Every solicitation costs 2.00 € (a fake, unrealistic figure used for the purpose of this exercise).
If the expected revenue I have predicted exceeds that figure of 2 €, I will recommend the charity
to solicit that individual (solicit = 1), since the expected profit is positive. If it is below 2 €, I will
recommend the charity not to solicit that individual (solicit = 0), since on average I expect a loss.

#### Process
1. Calibrate a discrete model (%) to predict the likelihood of donation (on individuals where calibration = 1)
2. Calibrate a continuous model (€) to predict the most likely donation amount in case of
donation (on the subset of individuals where donation = 1)
3. Apply both models to the prediction data (i.e., individuals where calibration = 0), and multiply
these predictions (% and €) to obtain expected (predicted) revenue if solicited.
4. If expected revenue is superior to 2.00 €, solicit (=1); otherwise do not (=0).

#### Methodology
1. Extract user features include reference, frequency, maxamount, avgamount and so on. 
2. Use XGBOOST to predict the probability that the user donates after being solicited.
3. Use Lasso Regression to predict the amount every user might pay if donate.
4. Obtain expected (predicted) revenue if solicited. If expected revenue is superior to 2.00 €, solicit (=1); otherwise do not (=0).

#### Result
The predictive result gets 215,191 EUR net profit, reach top 10% in the class.
