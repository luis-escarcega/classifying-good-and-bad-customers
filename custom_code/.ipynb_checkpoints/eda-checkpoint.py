import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_default_rate(df, target_column, record_date_column, figsize):
	mean_default = df.groupby(record_date_column).agg({target_column : "mean"})
	fig, ax = plt.subplots(figsize=figsize)
	ax.plot(mean_default)
	ax.tick_params(axis='x', rotation=90)
	ax.set(title="Default rate", xlabel = "Cohort", ylabel="Probability of default")
	plt.show()