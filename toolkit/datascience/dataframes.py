from math import log, floor
import pandas as pd
import numpy as np
from termcolor import colored

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 300)


# ---------------------------------------------------------------------------------------------------
def df_format_columns(df, columns, to_format):
	"""

	Args:
		df (pd.DataFrame):
		columns (list or str):
		to_format (str): [%, $]

	Returns:
		pd.DataFrame:
	"""

	if type(columns) != list:
		columns = [columns]

	# Formatters
	def _to_currency(x):
		return f'${x:.2f}'

	def _to_percentage(x):
		return f'{x:.2%}'


	if to_format == '%':
		func = _to_percentage
	else:
		func = _to_currency


	def _convert(x):
		try:
			return func(x)
		except Exception as e:
			return x

	for column in columns:
		df[column] = df[column].apply(lambda x: _convert(x) )

	return df


def df_humanize_columns_number_format(number, keep_numeric=False):
	"""

	Args:
		number: Numerical data only.
		keep_numeric (bool):
	Returns:
		float or str
	"""
	try:
		if type(number) not in [float, int, np.float, np.int]:
			return number
		elif abs(number) < 1:
			# return f'{number:.4f}'
			return round(number, 4)
		elif abs(number) < 1000:
			# return f'{number:.2f}'
			return round(number, 2)
		elif np.isnan(number):
			return number
		if keep_numeric:
			return round(number, 2)
		
		units = ['', 'K', 'M', 'B', 'T', 'P']
		k = 1000.0
		magnitude = int(floor(log(abs(number), k)))
		return '%.2f%s' % (number / k ** magnitude, units[magnitude])
	except Exception as e:
		# print(f'Number: {number} | Type: {type(number)} | Error: {e}')
		return number



def humanize_df_numbers(df, keep_numeric=False):
	"""
	
	Args:
		df (pd.DataFrame):
		keep_numeric (bool):

	Returns:
		pd.DataFrame:
	"""
	for col in df.columns.tolist():
		df[col] = df[col].map(lambda x: df_humanize_columns_number_format(x, keep_numeric=keep_numeric))

	return df


def df_clean_bad_chars(df):
	"""
	Cleans df from bad characters for saving to csv.

	Args:
		df (pd.DataFrame):

	Returns:
		pd.DataFrame:
	"""
	df.replace({',': '', '\n': '', '\r': ''}, regex=True, inplace=True)
	return df


def df_lowercase_columns(df):
	"""
	
	Args:
		df (pd.DataFrame):

	Returns:
		pd.DataFrame:
	"""
	rename = {}
	for col in df.columns.tolist():
		rename[str(col)] = str(col).lower()
	df.rename(rename, axis=1, inplace=True)
	return df


def df_print_cols_to_list(df):
	"""
	Prints an array to a list.

	Args:
		df pd.DataFrame:
	"""
	def _print_array_to_list(array):
		"""

		Args:
			array (list):
		"""
		text = '[\n'
		for col in array:
			text += f"'{col}', \n"
		text += ']'
		print(text)

	_print_array_to_list(df.columns.tolist())


def df_to_csv(df, file_path, line_terminator='\n', sep=',', index=False, remove_bad_chars=True):
	"""
	Save a df to a csv.

	Args:
		df (pd.DataFrame):
		file_path (str):
		line_terminator (str):
		sep (str):
		index (bool):
		remove_bad_chars (bool):
	"""

	if remove_bad_chars:
		df = df_clean_bad_chars(df)

	df.to_csv(file_path, line_terminator=line_terminator, sep=sep, index=index)


def df_filter_between(df, column, low=None, high=None, verbose=False):
	"""

	Args:
		df (pd.DataFrame):
		column (str or int):
		low (int or float):
		high (int or float):
		verbose (bool):
	Returns:
		pd.DataFrame:
	"""
	shape_0 = df.shape[0]

	if low is not None:
		df = df[(df[column].isna()) | (df[column] >= low)]

	if high is not None:
		df = df[(df[column].isna()) | (df[column] <= high)]

	shape_1 = df.shape[0]
	lost = shape_0 - shape_1
	if verbose:
		print(f'\nColumn: {column} \nLow: {low} | High: {high} | Rows lost: {lost:,} | {lost / shape_0:.2%}')
	return df

# ---------------------------------------------------------------------------------------------------

def df_print(df, index=True):
	"""
	Args:
		df (pd.DataFrame):
		index (bool):
	"""
	header = f'ROWS: {df.shape[0]:,} | COLUMNS: {df.shape[1]:,}'
	header += f'{" " * (250 - len(header))}'
	print(colored(f"{'-' * 250}", 'white', 'on_grey'))
	print(colored(header, 'white', on_color='on_grey'))
	text = df.to_string(justify='center', index=index)

	toggle = False
	lines = text.split('\n')
	for line in lines:
		line += (250 - len(line)) * ' '
		if toggle:
			print(colored(line, 'grey', 'on_white'))
		else:
			print(colored(line, 'white', on_color='on_grey'))

		toggle = not toggle
	print(colored(f"{'-' * 250}", 'white', 'on_grey'))


def df_null(df, print_df=True):
	"""
	Displays the null count for columns.

	Args:
		df (pd.DataFrame):
		print_df (bool): Print df or not

	Returns:
		pd.DataFrame:
	"""
	null_perc = (df.isnull().sum() / df.shape[0])
	null_perc.sort_values(inplace=True, ascending=False)

	null_perc = pd.DataFrame({
		'column': null_perc.index,
		'null_pct': null_perc.values
	})


	if print_df:
		null_perc = df_format_columns(null_perc, 'null_pct', '%')
		df_print(null_perc[['column', 'null_pct']], index=False)
	else:
		return null_perc


def df_corr(df, y_pred, sort_by='p_corr', print_df=True):
	"""

	Args:
		df (pd.DataFrame):
		y_pred (str):
		sort_by (str):
		print_df (bool): Print df or not.

	Returns:
		pd.DataFrame:
	"""

	p_corr = df.corrwith(df[y_pred], axis=0, method='pearson')
	k_corr = df.corrwith(df[y_pred], axis=0, method='kendall')
	s_corr = df.corrwith(df[y_pred], axis=0, method='spearman')

	df = pd.DataFrame({
		'column': p_corr.index,
		'p_corr': p_corr.values,
		'k_corr': k_corr.values,
		's_corr': s_corr.values,
	})

	df.sort_values(by=sort_by, inplace=True)
	if print_df:
		df = df_format_columns(df, ['p_corr', 'k_corr', 's_corr'], '%')
		df_print(df, index=False)
	else:
		return df


def df_pivot_unique_count(df, columns):
	"""
	Get total unique count for a specified field.

	Args:
		df (pd.DataFrame):
		columns (list): List of columns to group by.

	Returns:
		pd.DataFrame:
	"""
	df1 = df.pivot_table(index=columns, aggfunc='size', fill_value=0)

	indexes = df1.index.values
	dct = {}
	for col in columns:
		dct[col] = []

	for index in indexes:
		for col, val in zip(columns, index):
			dct[col].append(val)
	dct['count'] = df1.values
	return pd.DataFrame(dct)


def df_corr_null_count(df, y_col):
	"""

	Args:
		df (pd.DataFrame):
		y_col (str):

	Returns:
		pd.DataFrame:
	"""
	null_count = df_null(df, print_df=False)
	corr = df_corr(df, y_col, print_df=False)
	df = pd.merge(null_count, corr, on='column', how='left')
	return df


# ---------------------------------------------------------------------------------------------------


def df_describe(df, y_pred=None, x_cols=None, sort_by='', return_df=False):
	"""

	Args:
		df (pd.DataFrame):
		y_pred (str):
		x_cols (list):
		sort_by (str):
		return_df (bool):
	Returns:

	"""

	# ---------------------------------------------------------------------------------------------------
	#  Info
	rows = df.shape[0]

	counts = df.apply(lambda x: x.count())
	null_count = df.apply(lambda x: x.isnull().sum())
	null_percentage = (df.isnull().sum() / rows) * 100
	unique_count = df.apply(lambda x: x.unique().shape[0])
	unique_values = df.apply(lambda x: [x.unique()])

	min_val = df.min(numeric_only=True).apply(lambda x: df_humanize_columns_number_format(x))
	max_val = df.max(numeric_only=True).apply(lambda x: df_humanize_columns_number_format(x))
	mean = df.mean(numeric_only=True).apply(lambda x: df_humanize_columns_number_format(x))
	std = df.std(numeric_only=True).apply(lambda x: df_humanize_columns_number_format(x))
	skew = df.skew(numeric_only=True).apply(lambda x: df_humanize_columns_number_format(x))
	kurtosis = df.kurtosis(numeric_only=True).apply(lambda x: df_humanize_columns_number_format(x))

	# ---------------------------------------------------------------------------------------------------
	# Format
	df_columns = [
		counts,
		null_count,
		null_percentage,
		unique_count,
		min_val,
		max_val,
		mean,
		std,
		skew,
		kurtosis
	]
	df_column_names = [
		'Counts',
		'NullCount',
		'NullPercentage',
		'UniqueCount',
		'Min',
		'Max',
		'Mean',
		'STD',
		'Skew',
		'Kurtosis'
	]
	formatters = {
		'Counts': '{:,}'.format,
		'NullCount': '{:,}'.format,
		'NullPercentage': '{:,.2f}%'.format,
		'UniqueCount': '{:,}'.format,
	}

	if type(unique_values) == pd.Series:
		df_columns.insert(4, unique_values)
		df_column_names.insert(4, 'UniqueValues')

	# ---------------------------------------------------------------------------------------------------
	#  Y prediction

	p_corr_ttl, k_corr_ttl, s_corr_ttl = 0, 0, 0
	if y_pred:
		p_corr = df.corrwith(df[y_pred], axis=0, method='pearson')
		k_corr = df.corrwith(df[y_pred], axis=0, method='kendall')
		s_corr = df.corrwith(df[y_pred], axis=0, method='spearman')
		df_columns.append(p_corr)
		df_columns.append(k_corr)
		df_columns.append(s_corr)

		df_column_names.append('P_corr')
		df_column_names.append('K_corr')
		df_column_names.append('S_corr')

		formatters['P_corr'] = '{:,.2%}'.format
		formatters['K_corr'] = '{:,.2%}'.format
		formatters['S_corr'] = '{:,.2%}'.format

		if x_cols:
			for x_col in x_cols:
				p_corr_ttl += abs(p_corr[x_col])
				k_corr_ttl += abs(k_corr[x_col])
				s_corr_ttl += abs(s_corr[x_col])
		else:
			for col in df.columns:
				if col != y_pred and col in p_corr:
					if not np.isnan(p_corr[col]):
						p_corr_ttl += abs(p_corr[col])
					if not np.isnan(k_corr[col]):
						k_corr_ttl += abs(k_corr[col])
					if not np.isnan(s_corr[col]):
						s_corr_ttl += abs(s_corr[col])

	# ---------------------------------------------------------------------------------------------------
	#  Creates dataframe
	df1 = pd.concat(df_columns, axis=1, sort=False)
	df1.columns = df_column_names
	if type(unique_values) == pd.Series:
		df1['UniqueValues'] = np.where(df1['UniqueCount'] >= 5, '-', df1['UniqueValues'])
		def _clean(x):
			try:
				x = x[0]
				new = []
				for i in x:
					new.append(round(i, 3))
				return new
			except:
				return x
		df1['UniqueValues'] = df1['UniqueValues'].apply(lambda x: _clean(x))
	# ---------------------------------------------------------------------------------------------------

	def _print_text():
		if sort_by:
			df1.sort_values(by=sort_by, inplace=True)
		header = f'ROWS: {df.shape[0]:,} | COLUMNS: {df.shape[1]:,}'
		if y_pred:
			header += f' | P_CORR: {p_corr_ttl:.2%} | K_CORR: {k_corr_ttl:.2%} | S_CORR: {s_corr_ttl:.2%}'
		header += (250 - len(header)) * ' '

		print(colored(f"{'-' * 250}", 'white', 'on_grey'))
		print(colored(header, 'white', on_color='on_grey'))

		text = df1.to_string(formatters=formatters, justify='center')
		toggle = False
		lines = text.split('\n')
		for line in lines:
			attrs = []
			if x_cols:
				for x in x_cols:
					# if x in line:
					if x == line[:len(x)]:
						attrs.append('bold')
						line += f'    {x}'
						break

			line += (250 - len(line)) * ' '
			if toggle:
				print(colored(line, 'grey', 'on_white', attrs=attrs))
			else:
				print(colored(line, 'white', on_color='on_grey', attrs=attrs))

			toggle = not toggle
		print(colored(f"{'-' * 250}", 'white', 'on_grey'))

	_print_text()
	if return_df: return df1



def get_symbol(df, symbol):
	"""
	
	Args:
		df (pd.DataFrame):
		symbol (str):

	Returns:
		pd.DataFrame:
	"""
	return df[df['symbol'] == symbol]
# ---------------------------------------------------------------------------------------------------













