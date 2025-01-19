import pandas as pd
import numpy as np

from toolkit import tools
from toolkit.datascience import *
# from datascience.dataframes import *
from toolkit.logger import get_logger

log = get_logger()


# ---------------------------------------------------------------------------------------------------

def remove_outlier_rows(df, column, remove_perc=.02, verbose=True):
	"""
	Remove outlier rows from dataframe using its quartile range.

	Args:
		df (pd.DataFrame):
		column (str):
		remove_perc (float):
		verbose (boolean):

	Returns:
		pd.DataFrame:
	"""
	low = np.quantile(df[~df[column].isna()][column], remove_perc / 2)
	high = np.quantile(df[~df[column].isna()][column], 1 - remove_perc / 2)
	df = df_filter_between(df, column, low, high, verbose=verbose)
	if verbose:
		plot_histogram(df[column])

	return df


def remove_columns_by_corr(df, y_col, threshold, corr_calc='s_corr', bad_cols=None):
	"""
	Filter dataframe columns by it's correlation to y.

	Args:
		df (pd.DataFrame):
		y_col (str):
		threshold (float): Threshold for minimum correlation percentage.
		corr_calc (str): Correlation calculation to use ['p_corr', 'k_corr', 'p_corr']
		bad_cols (list):
	Returns:
		pd.DataFrame:
	"""
	corr = df_corr(df, y_col, print_df=False)
	corr = corr[abs(corr[corr_calc]) >= threshold]
	cols = corr['column'].tolist()
	if bad_cols is not None:
		cols = tools.subtract_arrays(cols, bad_cols)
		all_cols = bad_cols.copy()
		all_cols.extend(cols)
		cols = all_cols

	if len(cols) <= 1:
		print('All columns removed')
		return pd.DataFrame()
	else:
		return df[cols]



class Scaler:
	stats: pd.DataFrame

	def __init__(self, x_train, x_test):
		self.x_train = x_train
		self.x_test = x_test

		self._init_stats()
		self.normalize()

	def _init_stats(self):
		cols = self.x_train.columns.tolist()
		self.stats = self.x_train.describe()[cols]
		self.stats = self.stats.transpose()

	def normalize(self):
		def _norm(x):
			return (x - self.stats['mean']) / self.stats['std']

		self.x_train = _norm(self.x_train)
		self.x_test = _norm(self.x_test)



	def normalize_dataset(self, x):
		def _norm(x):
			return (x - self.stats['mean']) / self.stats['std']
		return _norm(x)


	def reverse_normalize(self):
		self.x_train = (self.x_train * self.stats['std']) + self.stats['mean']
		self.x_test = (self.x_test * self.stats['std']) + self.stats['mean']
