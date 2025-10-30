# GARCH MIDAS Filter

This code fits a GARCH MIDAS volatility model for an index and a set of macro variables.
This method has the benefit of using both a long-term variance fitted on macro variables and a GARCH(1,1) model, a well-known effective description of short-term volatility dynamics.
The method used for calibration is standard Quasi Maximum Likelihood with gaussian residuals.

## Data requirements:

The return time series for the index/asset considered have daily frequency. The object expects a pd.Series.
Macro time series can have monthly or quarterly frequencies. The code assumes that the macro series have dates at end of month.
Please note that macro series are always lagged by 1 period before fitting so there is no risk of "look ahead" bias even if macro data is published late in the period.  
For macro variables, the object expects a dictionary of tuples in this order: 
1) pd.Series
2) frequency as a string 'M' or 'Q'
3) description as a string
4) type of filter as a string: 'MAV', 'Constrained' or 'MIDAS'
5) number of lags as an integer

## References:

On the Economic Sources of Stock Market Volatility, Engle et al., [2008](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=971310)

A Regime-Switching GARCH-MIDAS Approach to Modelling Stock Market Volatility,Naumann, K.W.,[2019](https://thesis.eur.nl/pub/50191) 

Predicting the long-term stock market volatility: A GARCH-MIDAS model with variable selection, T. Fang et al., [2020](https://faculty.ucr.edu/~taelee/paper/2020%20JEF.pdf)
