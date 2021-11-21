"""Collection of utilities functions."""

import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from imblearn.over_sampling import SMOTE


def fetch_data(d_type):
    """
    Return train or validation data as pandas dataframe.
    in:
        d_type: string with the csv file to read
    
    """
    if d_type == "train":
        return (pd.read_csv("./data/train_set.csv"))
    elif d_type == "test":
        return (pd.read_csv("./data/test_set.csv"))

    elif d_type not in [
        "train",
        "test",
    ]:
        print("Enter a valid data type.")


def is_unique(s):
    """
    check if s is unique

    in:
        s - column of a df
        
    output: boolean telling if the column is unique

    """
    a = s.to_numpy() 
    return (a[0] == a).all()

def columns_without_variability(df):
    """
    

    Parameters
    ----------
    df : dataframe
        df you want to check.

    Returns
    -------
    to_drop : list
        list containing the columns without variability.

    """
    to_drop = []
    for column in df.columns:
        if is_unique(df[column]):
            to_drop.append(column)
    
    return to_drop

def eliminate_duplicates(df,columns_to_check):
    """
    
    Parameters
    ----------
    df : dataframe
        dataframe you want to check.
    columns_to_check : list
        list of the columns you want to check.

    Returns
    df :df without duplicates

    """
    '''checks for each column in columns_to_check, if there is a duplicate in the df, if so, drops all the duplicates'''

    duplicates = []
    dropped = 0
    for f in range(len(columns_to_check)-1):    
        for f2 in columns_to_check[f+1:]:
                    #print(inf_f[f],f2)
            try:
                df[columns_to_check[f]].shape
                df[f2]
            except KeyError:
                pass
            else:
                if df[columns_to_check[f]].equals(df[f2]):
                    duplicates.append(f2)
                    dropped += 1
                    df = df.drop([f2],axis = 1)
    print(dropped, ' dropped columns')
    
    return df

def drop_columns_without_variability(df):
    """
    Drop from df columns that have no variability in training data. Can be applied to train and test.

    Parameters
    ----------
    df : dataframe
        df to drop the  columns from.

    Returns
    -------
    df : dataframe
        df cleaned.

    """
    from config import columns_without_variablity
    
    if len(columns_without_variablity) ==0:
        columns_without_variablity = columns_without_variability(df)
        
        
    for col in columns_without_variablity:
        if col in df:
            df.drop(columns=[col], inplace=True)
    return df

def drop_columns_duplicates(df):
    """
    Drop from df columns that are duplicates in training data. Can be applied to train and test.

    Parameters
    ----------
    df : dataframe
        df to drop the  columns from.

    Returns
    -------
    df :dataframe
        df cleaned.

    """

    from config import duplicates_list

    if len(duplicates_list) == 0:
        df = eliminate_duplicates(df,df.columns)
    
    else:
        for col in duplicates_list:
            if col in df:
                df.drop(columns=[col], inplace=True)
    return df








def reduce_mem_usage(df):
    """
    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage.

    Parameters
    ----------
    df : dataframe
        df of you want to reduce memory usage.

    Returns
    -------
    df : dataframe
        new df.

    """
    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print(
        "Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)".format(
            start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem
        )
    )
    return df


def correlation(dataset, threshold):
    """
    

    Parameters
    ----------
    dataset : dataframe
        the dataframe you want to use.
    threshold : number 
        the threshold for which you want to drop a column.

    Returns
    -------
    col_corr : set
        all columns with a corr higher than the threshold.

    """
    col_corr = set()  # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (
                corr_matrix.columns[j] not in col_corr
            ):
                colname = corr_matrix.columns[i]  # getting the name of column
                print(
                    colname,
                    " correlated with",
                    corr_matrix.columns[j],
                    " corr: ",
                    corr_matrix.iloc[i, j],
                )
                col_corr.add(colname)
                # if colname in dataset.columns:
                # del dataset[colname] # deleting the column from the dataset
    return col_corr






    
def subset_data(df, shuffle_type, prcn=1, smote_os=0):
    """
    Decide what type of shuffle is applied to data, then oversamples minority class to hit ratio specified in smote_os.

    Parameters
    ----------
    df : dataframe
        df you want to modify.
    shuffle_type : string
        type of shuffle.
    prcn :  number, optional
        the % of the dataset you want to use. The default is 1.
    smote_os : number, optional
        sampling_strategy for smote. The default is 0.

    Returns
    -------
    df : dataframe
        df prepared.

    """


    if smote_os > 0:
        oversample = SMOTE(sampling_strategy=smote_os)
        X, y = oversample.fit_resample(
            df.drop(["target"], axis=1), df.target
        )
        df = X.copy()
        df["target"] = y

    if shuffle_type == "last":
        drop_train_idx = (
            df[df["target"] != 1]
            .loc[
                : int(len(df["target"]))
                - int(len(df[df["target"] != 1]) / prcn)
            ]
            .index
        )
        df = df.drop(df.index[drop_train_idx])

    if shuffle_type == "first":
        drop_train_idx = (
            df[df["target"] != 1]
            .loc[int(len(df[df["target"] != 1]) / prcn) :]
            .index
        )
        df = df.drop(df.index[drop_train_idx])

    if shuffle_type == "random":
        df = df.sample(int(len(df) * prcn))

    return df


def sub_template_creator(df):
    """creates the templated for the submission"""
    submission_template = pd.DataFrame(index=range(len(df)))
    submission_template["y_test_pred"] = 0

    return submission_template








def prepare_train_test_before_scoring(df, test):
    """
    creates train and test df

    Parameters
    ----------
    df : dataframe
        dataframe you want to use for training.
    test : dataframe
        test df.

    Returns
    -------
    X_train : dataframe
        df containing train features.
    y_train : dataframe
         df containing train features.
    X_test : dataframe
        df containing test features.

    """

    training = df.copy()
    labels = training["target"]  # 1.2%
    #test = test.drop(["id"], axis=1)
    for col in ["target"]:
        if col in training.columns:
            training.drop(columns=[col], inplace=True)

    X_train, y_train = training, labels
    X_test = test

    X_train.columns = [
        "".join(c if c.isalnum() else "_" for c in str(x)) for x in X_train.columns
    ]
    X_test.columns = [
        "".join(c if c.isalnum() else "_" for c in str(x)) for x in X_test.columns
    ]
    return X_train, y_train, X_test






                
                         
                
                
                
