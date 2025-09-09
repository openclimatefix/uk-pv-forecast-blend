The data in the `model_horizon_mae_scores.csv` file is used to optimise the model blends based on
on the expected accuracy of each model for each forecast horizon, and on the initialisation times 
available for each model in production.

There is a column in the csv for each model we would like to consider in the blend. The column names 
must match the names which the model results are saved under in the database. Each row in the table
is for a different forecast horizon (ascending timedeltas). 

The values are the normalised mean absolute error (NMAE) expected for each model for each forecast 
horizon. In the table, where a model does not predict out to a given horizon its NMAE is set
to NaN. For example, the pvnet-dat-ahead model goes out to 36 hours, but the intraday models only go 
out to 8 hours. Therefore, all the NMAEs for the intraday models above 8 hours are set to NaN.

The NMAE values could be taken from the validation results from training a model, from a backtest, 
or even from the live production performance. It is important that the NMAE results for each model 
be comparable to each other or else suboptimal blends may be produced. For example, if the NMAE 
values got two models came from their validation results on two different sets of presaved samples 
then we cannot in general compare them. In practice, we are likely to relax this limitation so long
as it produces reasonable blend suggestions.

Note that the blend does not take into account differing model performance for GSP or national 
predictions. Only a single score is used.

For the current scores, we have taken the NMAE values for each model from the GSP-level validation 
results for the validation period (2022). This is true for all the intraday models, and for the 
pvnet-day-ahead model. All models were assessed on the exact same init-timesand are
fully compatible. 

The National_xg model NMAEs are all set to 0.05. These were not measured and instead were set 
as a value higher than all NMAEs for the other values. From observation we know that the National_xg
model underperforms PVNet so we only want to use it in the blend if we have run out of PVNet 
predictions.