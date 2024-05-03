import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from optbinning import OptimalBinning
from statsmodels.stats.proportion import proportions_ztest
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import chi2_contingency

def _is_null_info_meaningful(y, predictor, significance):
    null_mask_predictor = np.isnan(predictor)
    
    y_null = y[null_mask_predictor]
    y_not_null = y[~null_mask_predictor]

    count = np.array([np.sum(y_null), np.sum(y_not_null)])
    nobs = np.array([len(y_null), len(y_not_null)])

    stat, pval = proportions_ztest(count, nobs)

    return round(pval, 10), pval < significance

def _null_info_study(X, y, significance):
	predictor_columns = X.columns.tolist()
	null_info_study = []

	for column in predictor_columns:
	    predictor = X[column]
	    pval, test = _is_null_info_meaningful(y, predictor, significance)
	    null_info_study.append((column, pval, test))

	df_null_info_study = pd.DataFrame(null_info_study, columns=["predictor", "p_value_ztest", "is_null_info_meaningful"])
	return df_null_info_study

def null_info_study(X, y, significance=0.05, figsize=(10,7), top=None):
	
	df_null_info_study_orig = _null_info_study(X, y, significance)
	df_null_info_study = df_null_info_study_orig.sort_values("p_value_ztest")
	
	x = df_null_info_study["predictor"][:top]
	y = df_null_info_study["p_value_ztest"][:top]

	fig, ax = plt.subplots(figsize=figsize)
	ax.bar(x, y)
	ax.tick_params(axis='x', rotation=90)
	ax.axhline(y = significance, color = 'r', linestyle = '--') 
	ax.set(ylabel="p-value of Porportions Z-test")
	plt.show()

	return df_null_info_study_orig

def _get_gini_coefficient(y, predictor):
    mask_no_null = ~np.isnan(predictor)
    predictor_no_null = predictor[mask_no_null]
    y_no_null = y[mask_no_null]
    gini = 2*roc_auc_score(y_no_null, predictor_no_null) - 1
    return gini

def _compute_confidence_interval(y, predictor, n_resamples, confidence_level, random_state):
    rnd = np.random.RandomState(random_state)
    n_sample = len(y)
    significance = 1 - confidence_level
    p, q = significance/2, 1 - (significance/2)

    bootstraped_ginis = []
    
    for i in range(n_resamples): 
        index_bootstrap = rnd.choice(n_sample, n_sample)
        y_bootstrap = y[index_bootstrap]
        predictor_bootstrap = predictor[index_bootstrap]
        gini = _get_gini_coefficient(y_bootstrap, predictor_bootstrap)
        bootstraped_ginis.append(gini)

    ci_lower, ci_upper = np.quantile(bootstraped_ginis, [p, q])

    return ci_lower, ci_upper

def gini_coefficient_bootstrapping(X, y, n_resamples=999, confidence_level=0.95, random_state=42):
	predictor_columns = X.columns.tolist()
	n_predictors = len(predictor_columns)
	n_prints = 10
	step_prints = int(n_predictors / n_prints)
	ginis_data = []
	
	for idx, column in enumerate(predictor_columns):
		predictor = X[column]
		ci_lower, ci_upper = _compute_confidence_interval(y, predictor, n_resamples, confidence_level, random_state)
		ginis_data.append((column, ci_lower, ci_upper))
		
		if ((idx+1) % step_prints == 0):
			print(f"Completed: {(idx+1) / n_predictors * 100 : .2f}%.")
	
	df_ginis = pd.DataFrame(ginis_data, columns=["predictor", "gini_lower_bound", "gini_upper_bound"])
	
	return df_ginis

class OptbinTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, cols_drop, cols_null_indicator, cols_map_and_sense):
        self.cols_drop = cols_drop
        self.cols_null_indicator = cols_null_indicator
        self.cols_map_and_sense = cols_map_and_sense
    
    def fit(self, X, y):
        dict_of_optbins = {}
        
        for column, sense in self.cols_map_and_sense: 
            predictor = X[column].values
            optb = OptimalBinning(name=column, dtype="numerical", solver="cp", monotonic_trend=sense)
            optb.fit(predictor, y)
            dict_of_optbins[column] = optb
        
        self.dict_of_optbins = dict_of_optbins

        return self

    def transform(self, X, y=None):
        # dropping columns that are not needed
        X_transformed = X.drop(columns=self.cols_drop).copy()
        # taking null indicator variables
        X_transformed[self.cols_null_indicator] = X_transformed[self.cols_null_indicator].isna().astype(int)
        # transforming the rest of columns
        
        for column, sense in self.cols_map_and_sense:
            optb = self.dict_of_optbins[column]
            X_transformed[column] = optb.transform(X_transformed[column], metric="event_rate", metric_missing="empirical")
        
        return X_transformed

def test_stability(df, target_column, record_date_column, predictors_and_sense, significance=0.05):

	stability_results = []
	
	record_date = df[record_date_column]
	y = df[target_column]
	
	for column, sense in predictors_and_sense:
		predictor = df[column]

		if sense is not None:
			optb = OptimalBinning(name=column, dtype="numerical", solver="cp", monotonic_trend=sense, max_n_prebins=4)
			optb.fit(predictor, y)
			predictor_indices = optb.transform(predictor, metric="indices")
		else:
			predictor_indices = predictor # it the null indicator variable
		
		df_predictor_indices = pd.DataFrame({
			record_date_column : record_date, 
			column : predictor_indices, 
			target_column : y})
		
		df_predictor_indices[f"{column}-{target_column}"] = \
			df_predictor_indices[column].astype(str) + df_predictor_indices[target_column].astype(str)

		df_predictor_groupped = df_predictor_indices.groupby([record_date_column, f"{column}-{target_column}"]).size().reset_index()
		df_predictor_pivoted = df_predictor_groupped.pivot(index=record_date_column, columns=f"{column}-{target_column}", values=0)
		df_predictor_pivoted.fillna(0, inplace=True)
		
		contigency_table = df_predictor_pivoted.values
		res = chi2_contingency(contigency_table)
		
		stability_results.append((column, res.statistic, res.pvalue, res.pvalue < significance))

	df_stability_results = pd.DataFrame(stability_results, columns=["predictor", "chi_square_statistic", "p_value_chi_square", "is_unstable_variable"])
	
	return df_stability_results
























