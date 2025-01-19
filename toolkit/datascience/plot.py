# Matplot lib documentation
# https://matplotlib.org/3.1.1/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py
# Diagram: https://matplotlib.org/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py

# Figure	- Holds all of the charts
# Axes 		- A plot. A figure can contain more than one axes.
# Axis		- Number line like object. They generate ticks, tick labels etc.
# ---------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from scipy.stats import norm, skew

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pio.renderers.default = "browser"
sns.set_style('darkgrid')

# ---------------------------------------------------------------------------------------------------
# Matplot

def plot_histogram(array, title=''):
	"""
	Plot a plot_histogram from an array.

	Args:
		array (list or pd.core.series.Series):
		title (str):
	"""

	if type(array) == pd.core.series.Series:
		title = array.name
		na_count = array[array.isna()].shape[0]
		if na_count != 0:
			title += f' | Percent NA: {na_count / array.shape[0]:.2%}'
		array = array.tolist()

	plt.figure(figsize=(12, 8))

	n, bins, patches = plt.hist(array, weights=np.ones(len(array)) / len(array), color='b')
	ax = plt.gca()

	# ---------------------------------------------------------------------------------------------------
	#  Get labels
	labels0 = []

	for patch in patches:
		height = patch.get_height()
		labels0.append(f'{height:,.2%} \n{math.ceil(height * len(array)):,}')

	ranges = bins.copy().tolist()
	labels = []
	for label in labels0:
		start = ranges.pop(0)
		labels.append(label + f'\n ({start:.2f} \n, {ranges[0]:,.2f})')

	# ---------------------------------------------------------------------------------------------------
	for patch, label in zip(patches, labels):
		ax.text(
			patch.get_x() + patch.get_width() / 2,
			patch.get_height(),
			label,
			ha='center',
			va='bottom'
		)

	# ---------------------------------------------------------------------------------------------------

	plt.hist(array, weights=np.ones(len(array)) / len(array))
	plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
	plt.title(title)

	plt.show()


# ---------------------------------------------------------------------------------------------------
# Plotly

def plot_df(df, y, x='', title='', group_by=''):
	"""
	Plot a line chart in plotly.

	Args:
		df (pd.DataFrame):
		y (str):
		x (str):
		title (str):
		group_by (str):
	"""
	# https://plotly.com/python/hover-text-and-formatting/#customizing-hover-text-with-plotly-express
	if not x:
		x = df.index
	cols = df.columns.tolist()
	
	if group_by != '':
		fig = px.line(df, x=x, y=y, title=title, color=group_by, hover_data=cols)
	else:
		fig = px.line(df, x=x, y=y, title=title, hover_data=cols)

	fig.show()


def plot_df_double_y(df, y1, y2, x=''):
	"""
	
	Args:
		df (pd.DataFrame):
		y1 (str):
		y2 (str):
		x (str):
	"""
	# https://stackoverflow.com/questions/55713856/pandas-plotly-how-to-access-data-columns-in-the-hover-text-that-are-not-used
	if not x:
		x = df.index
	fig = make_subplots(specs=[[{"secondary_y": True}]])
	fig.add_trace(go.Scatter(x=x, y=df[y1], name=y1), secondary_y=False)
	fig.add_trace(go.Scatter(x=x, y=df[y2], name=y2), secondary_y=True)
	fig.update_yaxes(title_text=y1, secondary_y=False)
	fig.update_yaxes(title_text=y2, secondary_y=True)
	
	fig.show()


# ---------------------------------------------------------------------------------------------------
# Seaborn

def plot_distribution(series):
	sns.distplot(series , fit=norm)
	plt.show()









