
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/00_brds.ipynb

from imports import *


#package all into a function
def get_data(file_id):
    import gdown
    url = f'https://drive.google.com/uc?id={file_id}'
    zip_file = 'data/data.zip'
    gdown.download(url, zip_file, quiet=False)

    import  zipfile
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall('data/unzipped_data')

    users_raw = pd.read_csv('data/unzipped_data/nsds_users.csv')
    purchases_raw = pd.read_csv('data/unzipped_data/nsds_purchases.csv')

    return  users_raw, purchases_raw

class DataframeFunctionTransformer():
    """Creates a pandas Dataframe transformer from a function"""
    def __init__(self, func, **func_params):
        self.func = func
        self.func_params = func_params

    def transform(self, input_df, **transform_params):
        return self.func(input_df.copy(), **self.func_params)

    def fit(self, X, y=None, **fit_params):
        return self

def select_columns_(df, cols=None):
    return df[cols]

def timify_(df, cols):
    for c in cols: df[c]= pd.to_datetime(df[c])
    return df

def add_date_(df, cols):
    for c in cols: df[f'{c}_date']= pd.to_datetime(df[c]).dt.date
    return df

def categorify_(df, cols):
    for c in cols: df[c]= df[c].astype('object')
    return df

def purchases_to_rfm_(df):
    """
    wrapper to the function summary_data_from_transaction_data
    https://github.com/CamDavidsonPilon/lifetimes/blob/0a0a84fe4b10fff0bdaa6a6020d930c8dc6aee2d/lifetimes/utils.py#L230
    """
    from lifetimes.utils import  summary_data_from_transaction_data
    data = summary_data_from_transaction_data(
                                transactions = df,
                                customer_id_col = "user_id",
                                datetime_col = "purchased_at_date",
                                monetary_value_col = "value",
                                freq = "D").reset_index()
    return data



class FramesLeftMerger(BaseEstimator,TransformerMixin):
    """Creates a pandas Dataframe from a two pipelines and left joins them"""
    def __init__(self, pipe1, pipe2):
        self.pipe1 = pipe1
        self.pipe2 = pipe2

    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        df1, df2 = X[0], X[1]
        df1 = self.pipe1.transform(df1.copy())
        df2 = self.pipe2.transform(df2.copy())
        merge = df1.merge(df2, how='left', on='user_id')
        merge[df2.columns] = merge[df2.columns].fillna(0)
        return merge

preprocess_purchases = Pipeline(
    steps=[
        ("timify", DataframeFunctionTransformer(timify_ , cols=['purchased_at'])),
        ("add_date", DataframeFunctionTransformer(add_date_ , cols=['purchased_at'])),
        ("rfm", DataframeFunctionTransformer(purchases_to_rfm_)),
    ]
)


users_num_features = ['birthyear', 'dx_0', 'dx_1', 'dx_2', 'dx_3', 'gx', 'im']
users_cat_features = ['gender', 'maildomain', 'region', 'orig_1', 'orig_2', 'utm_src', 'utm_med', 'utm_cpg', 'channel']
select_cols = ['user_id', 'created_at'] + users_cat_features + users_num_features


preprocess_users = Pipeline(
    steps=[
        ("select_cols", DataframeFunctionTransformer(select_columns_ , cols=select_cols)),
        ("timify", DataframeFunctionTransformer(timify_ , cols=['created_at'])),
        ("categorify",DataframeFunctionTransformer(categorify_ , cols=users_cat_features))
    ]
)

full_preprocess = FramesLeftMerger(preprocess_users, preprocess_purchases)

class BetaGeoFitterTransformer(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.bgf = BetaGeoFitter(penalizer_coef=1e-06)

    def fit(self,X, y=None, **fit_params):
        self.bgf.fit(
            frequency = X["frequency"],
            recency = X["recency"],
            T = X["T"],
            weights = None,
            verbose = True,
            tol = 1e-06)
        return self

    def transform(self,X, y=None,  **transform_params):
        X['bgf'] = self.bgf.conditional_expected_number_of_purchases_up_to_time(90, X["frequency"], X["recency"], X["T"])
        return X


class GammaGammaFitterTransformer(BaseEstimator,TransformerMixin):
    def __init__(self):
        self.ggf = GammaGammaFitter(penalizer_coef = 1e-06)

    def fit(self,X, y=None, **fit_params):
        X_v = X[X.monetary_value>0]
        self.ggf.fit(
            frequency = X_v["frequency"],
            monetary_value = X_v["monetary_value"],
            weights = None,
            verbose = True,
            tol = 1e-06)
        return self

    def transform(self,X, y=None,  **transform_params):
        preds= self.ggf.conditional_expected_average_profit(X["frequency"], X["monetary_value"])
        X.loc[X.monetary_value>0 , 'ggf'] = preds
        X['ggf'] = X['ggf'].fillna(0)
        return X



def group_rare_categories(df, limit= 30, cols=None):
    df_c = df.copy()
    for c in cols:
        n = min(df_c[c].nunique() - 2, limit)
        df_c.loc[~df_c[c].isin( df_c[c].value_counts().iloc[:n].index ), c] = -1
    return df_c

# cat_cols = X.select_dtypes(exclude="number").columns[1:]
cat_cols = ['gender', 'maildomain', 'region', 'orig_1', 'orig_2', 'utm_src','utm_med', 'utm_cpg', 'channel']

extra_features_pipe = Pipeline(
    steps=[
            ("clean_categories", DataframeFunctionTransformer(group_rare_categories, cols=cat_cols)),
            ("betageo", BetaGeoFitterTransformer()),
            ("gammagamma", GammaGammaFitterTransformer()),
           ]
)

final_preprocess_pipeline = Pipeline(steps=[
    ('full_preprocess', full_preprocess),
    ('extra_features_pipe', extra_features_pipe),
])