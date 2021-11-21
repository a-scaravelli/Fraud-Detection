

import os
import pandas as pd
import seaborn as sns
import gc
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix  # da implementare
from datetime import datetime
from imblearn.over_sampling import SMOTE


current_dir = os.getcwd()
main_path = os.path.dirname(current_dir)
os.chdir(main_path)

from src.models import model_train
from src.utils import (
    reduce_mem_usage,
    sub_template_creator,
    prepare_train_test_before_scoring,
    fetch_data,
    subset_data,
    drop_columns_without_variability,
    drop_columns_duplicates,
    correlation
)


gc.collect() #21

# Importing data
training = fetch_data("train")
# training = pd.read_csv(main_path + r'\data\train_set.csv')
        
test = fetch_data("test")
# test = pd.read_csv(main_path + '\\data\\test_set.csv')

gc.collect() #141



#preprocessing
training = reduce_mem_usage(training)
training = drop_columns_without_variability(training)
training = drop_columns_duplicates(training)
correlated = correlation(training,0.9)
training = training.drop(correlated,axis = 1)


test = reduce_mem_usage(test)
test = drop_columns_without_variability(test)
test = drop_columns_duplicates(test)
test = test.drop(correlated,axis = 1)

gc.collect() #121

# easy way to filter non null targets
training = training[training['target'].notnull()]

# balance the dataset
training = subset_data(training, "random", prcn=1, smote_os=0.1)


print("train shape: ", training.shape, " - test shape: ", test.shape)

# defining predictions dataframe
submission_template = sub_template_creator(test)

# creating X_train,y_train, X_test
X_train, y_train, X_test = prepare_train_test_before_scoring(training, test)


models = ['lightgbm','xgboost','catboost']

models_compare = {}
for mdl in models:
    
    n_fold = 10  
    
    folds = KFold(n_fold)
    ROC_Avg = 0
    submission = submission_template.copy()
    start_time = datetime.now()
    mdl_list = []
    
    # train the model
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train)):
        print(fold_n)
        now = datetime.now()
    
        X_train_, X_valid = X_train.iloc[train_index], X_train.iloc[valid_index]
        y_train_, y_valid = y_train.iloc[train_index], y_train.iloc[valid_index]
    
    
        model_type = mdl
    
        pred_model, X_valid, y_valid = model_train(
            model_type, X_train_, X_valid, y_train_, y_valid
        )
    
        pred = pred_model.predict_proba(X_test)[:, 1]
        val = pred_model.predict_proba(X_valid)[:, 1]
        print("finish pred")
        del X_valid
        ROC_auc = roc_auc_score(y_valid, val)
        print("ROC accuracy: {}".format(ROC_auc))
        
        ROC_Avg = ROC_Avg + ROC_auc
        del val, y_valid
        submission["y_test_pred"] = submission["y_test_pred"] + pred / n_fold
        mdl_list.append(pred_model)
        del pred
        model_time = datetime.now() - now
        total_exp_time = model_time * n_fold
        current_time = datetime.now() - start_time
        print(
            "The current model took in total",
            model_time,
            "\n Still missing, this time:",
            str(total_exp_time - current_time),
        )
        gc.collect()
    
    
    print("\nAverage ROC for the model  ",mdl," is: ", ROC_Avg / n_fold)
    print("Total time to train the models is: ", current_time)
    models_compare[mdl] =  ROC_Avg / n_fold


print(models_compare)


#create submission
submission.to_csv('predictions.csv', header=False, index=False)



