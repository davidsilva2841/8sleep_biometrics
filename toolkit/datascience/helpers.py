import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------------------------------

def get_results(y_pred, y_test):
	"""

	Args:
		y_pred: np.ndarray or pd.series
		y_test: np.ndarray or pd.series

	Returns:
		pd.DataFrame:
	"""
	def _get_list(x):
		if type(x) == np.ndarray:
			return x.flatten()
		else:
			return x.tolist()

	return pd.DataFrame({
		'y_pred': _get_list(y_pred),
		'y_test': _get_list(y_test)
	})


def calc_positive(df_pred, label='', verbose=True):
	"""

	Args:
		df_pred (pd.DataFrame):
		label (str):
		verbose (bool):

	Returns:
		(pd.DataFrame, float):
	"""
	df_pred['y_pred_pos'] = np.sign(df_pred['y_pred'])
	df_pred['y_test_pos'] = np.sign(df_pred['y_test'])
	correct = df_pred[df_pred['y_pred_pos'] == df_pred['y_test_pos']].shape[0] / df_pred.shape[0]
	if verbose:
		print(f'{label}{correct:.2%} Correctly predicted positive or negative')
	return df_pred, correct











