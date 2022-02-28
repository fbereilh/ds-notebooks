#installs
# %pip install pycaret --quiet
# %pip install lifetimes --quiet
# %pip install gdown --quiet
# %pip install zipfile --quiet
# %pip install pyyaml==5.4.1
# !conda install scikit-learn=0.23.2 -y
#%pip freeze -> requirements.txt


# imports
from datetime import timedelta
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="darkgrid")
sns.set_context("notebook")

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.cluster import KMeans
from sklearn import model_selection

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import \
    calibration_and_holdout_data, \
    summary_data_from_transaction_data, \
    calculate_alive_path


from lifetimes.plotting import \
    plot_frequency_recency_matrix, \
    plot_probability_alive_matrix, \
    plot_period_transactions, \
    plot_history_alive, \
    plot_cumulative_transactions, \
    plot_calibration_purchases_vs_holdout_purchases, \
    plot_transaction_rate_heterogeneity, \
    plot_dropout_rate_heterogeneity

import dill


#set up
pd.options.display.float_format = '{:,.2f}'.format