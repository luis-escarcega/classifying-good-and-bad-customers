import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

def plot_default_rate(df, target_column, record_date_column, figsize, df_str=None, df_contrast=None, df_contrast_str=None):
	
	df_default_rate_summary = (
		df
		.groupby(record_date_column)
		.agg(
			default_rate = (target_column, "mean"),
			counting = (target_column, len)
		).reset_index()
	)

	if df_contrast is not None:
		df_contrast_default_rate_summary = (
			df_contrast
			.groupby(record_date_column)
			.agg(default_rate = (target_column, "mean"))
		)
	else:
		df_contrast_default_rate_summary = None

	date = df_default_rate_summary[record_date_column].values
	default_rate = df_default_rate_summary["default_rate"].values
	counting = df_default_rate_summary["counting"].values
	
	fig, ax0 = plt.subplots(figsize=figsize)
	ax0.plot(date, default_rate, label=df_str)
	ax0.tick_params(axis='x', rotation=90)
	ax0.set(title="Default rate", xlabel = "Cohort", ylabel="Probability of default")

	if df_contrast_default_rate_summary is not None:
		ax0.plot(df_contrast_default_rate_summary, label=df_contrast_str, ls="--")

	ax0.legend()
	
	ax1 = ax0.twinx()
	ax1.bar(date, counting, alpha=0.25)
	ax1.set(ylabel="Total number of clients")
	
	plt.show()

	counting_array = df_default_rate_summary["counting"].values
	reversed_counting_array = counting_array[::-1]
	cumulated_counting = reversed_counting_array.cumsum() / reversed_counting_array.sum()
	reversed_cumulated_counting = cumulated_counting[::-1]
	df_default_rate_summary["reversed_cumulated_counting"] = reversed_cumulated_counting

	return df_default_rate_summary