import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_default_rate(df, target_column, record_date_column, figsize):
	
	default_rate_summary = (
		df
		.groupby(record_date_column)
		.agg(
			default_rate = (target_column, "mean"),
			counting = (target_column, len)
		).reset_index()
	)

	date = default_rate_summary[record_date_column].values
	default_rate = default_rate_summary["default_rate"].values
	counting = default_rate_summary["counting"].values
	
	fig, ax0 = plt.subplots(figsize=figsize)
	ax0.plot(date, default_rate)
	ax0.tick_params(axis='x', rotation=90)
	ax0.set(title="Default rate", xlabel = "Cohort", ylabel="Probability of default")
	
	ax1 = ax0.twinx()
	ax1.bar(date, counting, alpha=0.25)
	ax1.set(ylabel="Total number of clients")
	
	plt.show()

	counting_array = default_rate_summary["counting"].values
	reversed_counting_array = counting_array[::-1]
	cumulated_counting = reversed_counting_array.cumsum() / reversed_counting_array.sum()
	reversed_cumulated_counting = cumulated_counting[::-1]
	default_rate_summary["reversed_cumulated_counting"] = reversed_cumulated_counting

	return default_rate_summary