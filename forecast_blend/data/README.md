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

For the current scores, we have taken the NMAE values for each model from the national results of a
2025 backtest using archive data. The NMAE-horizons curves for the model are calculated for an
identical set of t0 times. The exact model hashes used for these NMAE curves can be found in 
uk-pvnet-app v3.0.1.
