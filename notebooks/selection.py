from sklearn.model_selection import train_test_split
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

def get_dataset(path: str, one_hot_encode = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    df = df.drop(columns=["RefId", "PurchDate", "VehYear", "Trim", "Color",
             "WheelTypeID", "WheelType", "Nationality", "Size",
             "TopThreeAmericanName", "BYRNO", "VNST", "WarrantyCost"])
 
    X = df.drop(columns=["IsBadBuy"]) 
    y = df[["IsBadBuy"]]
    
    if one_hot_encode:
        cat_cols = []
        for i in X.columns:
            tpe = X[i].dtype
            if tpe not in ["float64", "int64"] and tpe == "object":
                cat_cols.append(i)

        trans = [(i+"_OHE", OneHotEncoder(handle_unknown='ignore'), [i]) for i in cat_cols]

        cols_w_null = X.columns[X.isnull().any()]

        num_impute_cols=[]
        cat_impute_cols=[]
        for i in cols_w_null:
            if X[i].dtype == 'object':
                cat_impute_cols.append(i)
            else:
                num_impute_cols.append(i)

        num_imputer = [(i, SimpleImputer(strategy='mean'), [i]) for i in num_impute_cols]
        cat_imputer = [(i, SimpleImputer(strategy='most_frequent'), [i]) for i in cat_impute_cols]


        if len(cat_imputer)> 0:
            cat_imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            X[cat_impute_cols] = pd.DataFrame(cat_imp.fit_transform(X[cat_impute_cols]), columns = cat_imp.get_feature_names_out())
        if len(num_imputer)> 0:
            num_imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            X[num_impute_cols] = pd.DataFrame(num_imp.fit_transform(X[num_impute_cols]), columns = num_imp.get_feature_names_out())

        # Create the column transformer
        transformer2 = ColumnTransformer(transformers=trans, remainder='passthrough', verbose_feature_names_out=True)
        # Fit the transformer to the data and transform the data
        one_hot_data = transformer2.fit_transform(X)
        X = pd.DataFrame(one_hot_data.todense(), columns=transformer2.get_feature_names_out())

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=42)
    return X_train, X_test, Y_train, Y_test
    