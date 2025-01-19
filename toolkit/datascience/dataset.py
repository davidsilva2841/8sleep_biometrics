import pandas as pd

from toolkit.datascience import *
from .preprocessing import Scaler
from toolkit import tools
from toolkit.logger import get_logger

log = get_logger()


# ---------------------------------------------------------------------------------------------------

class Data(Scaler):
	scaler: Scaler
	df: pd.DataFrame

	df_train: pd.DataFrame
	df_test: pd.DataFrame
	x_train: pd.DataFrame
	y_train: pd.DataFrame
	x_test: pd.DataFrame
	y_test: pd.DataFrame
	train_results: pd.DataFrame
	test_results: pd.DataFrame
	results: pd.DataFrame

	y_col: str
	x_cols: list
	xy_cols: list
	bad_cols: list
	test_size: float
	test_correct: float

	def __init__(self, df, y_col, x_cols=None, bad_cols=None, test_size=0.2, normalize=True):
		"""

		Args:
			df (pd.DataFrame):
			y_col (str):
			x_cols (list):
			bad_cols (list):
			test_size (float):
		"""

		self.df = df
		self.y_col = y_col
		self.x_cols = x_cols
		self.bad_cols = bad_cols
		self.test_size = test_size

		self._init_cols()
		self.split_train_test()
		if normalize:
			super(Data, self).__init__(self.x_train, self.x_test)


	def _init_cols(self):
		if not self.bad_cols and not self.x_cols:
			self.x_cols = tools.subtract_arrays(self.df.columns.tolist(), [self.y_col])
		elif self.bad_cols:

			self.xy_cols = tools.subtract_arrays(self.df.columns.tolist(), self.bad_cols)
			self.xy_cols.append(self.y_col)
			if not self.x_cols:
				self.x_cols = tools.subtract_arrays(self.xy_cols, [self.y_col])
		elif self.x_cols:
			self.bad_cols = tools.subtract_arrays(self.df.columns.tolist(), self.x_cols)
			self.xy_cols = self.x_cols.copy()
			if self.y_col not in self.xy_cols:
				self.xy_cols.append(self.y_col)
			else:
				log.error('y_col is in x_cols')


	def split_train_test(self):
		self.df_train = self.df.sample(frac=1 - self.test_size, random_state=0)
		self.df_test = self.df.drop(self.df_train.index)

		self.x_train = self.df_train[self.x_cols]
		self.y_train = self.df_train[self.y_col]

		self.x_test = self.df_test[self.x_cols]
		self.y_test = self.df_test[self.y_col]






	def join_preds(self, train_pred, test_pred):
		def _join(df, pred):
			df = df.reset_index(drop=True)
			df = pd.concat([df, pred], axis=1)

			cols = self.bad_cols.copy()
			cols.extend(['y_pred', 'y_test'])

			if 'y_pred_pos' in pred.columns.tolist():
				cols.extend(['y_pred_pos', 'y_test_pos'])

			for col in self.df.columns.tolist():
				if col not in cols:
					cols.append(col)
			return df[cols]


		self.train_results = _join(self.df_train, train_pred)
		self.test_results = _join(self.df_test, test_pred)
		self.results = self.train_results.append(self.test_results)
		self.results.sort_values(by=['symbol', 'year'], inplace=True)













