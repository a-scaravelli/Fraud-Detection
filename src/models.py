

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

import lightgbm as lgb

import xgboost as xgb
import catboost

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def model_train(model_type, X_train_, X_valid, y_train_, y_valid):
    """ lightgbm, xgboost, catboost, randomforest"""



    if model_type == "lightgbm":
        lgbclf = lgb.LGBMClassifier(
            num_leaves=256,  # was 512 - default 31
            n_estimators=512,  # default 100 was 512
            max_depth=9,  # default -1, was 9
            learning_rate=0.05,  # default 0.1
#            feature_fraction=0.4,  # default 1 was 0.4,
#            bagging_fraction=0.4,  # default 1 was 0.4, # subsample by row
            metric="binary_logloss",  # binary_logloss auc
            boosting_type="gbdt",  # goss # dart --> speed: goss>gbdt>dart
#            lambda_l1=0.3,  # default 0 - 0.4
#            lambda_l2=0.4,  # default 0 - 0.6
#            scale_pos_weight=8,  # defualt 1
        )

        lgbclf.fit(X_train_, y_train_)
        pred_model = lgbclf
        del lgbclf

    elif model_type == "xgboost":
        xgbclf = xgb.XGBClassifier(
                    num_leaves=512,
                    n_estimators=512,
                    max_depth = 9,
                    learning_rate=0.05,
#                    feature_fraction=0.4,
#                    bagging_fraction=0.4,
#                    subsample=0.85,
                    metric="binary_logloss",  # binary_logloss
                    colsample_bytree=0.85,
                    boosting_type="gbdt",  # goss # dart --> speed: goss>gbdt>dart
#                    reg_alpha=0.4,
#                    reg_lamdba=0.6,
#                    scale_pos_weight=82.9,
                )
        xgbclf.fit(X_train_, y_train_)
        pred_model = xgbclf
        del xgbclf

    elif model_type == "catboost":
        ycopy = y_train_.copy()
        ycopy = ycopy.astype(float)
        X_train_1, X_valid_1, y_train_1, y_valid_1 = train_test_split(
            X_train_, ycopy.values.flatten(), test_size=0.05
        )
        params = {
            "loss_function": "Logloss",  # objective function
            "eval_metric": "AUC",  # metric
            "verbose": 200,  # output to stdout info about training process every 200 iterations
        }
        catclf = catboost.CatBoostClassifier(**params)
        catclf.fit(
            X_train_1,
            y_train_1,  # data to train on (required parameters, unless we provide X as a pool object, will be shown below)
            eval_set=(X_valid_1, y_valid_1),  # data to validate on
            use_best_model=True,  # True if we don't want to save trees created after iteration with the best validation score
            plot=True,  # True for visualization of the training process (it is not shown in a published kernel - try executing this code)
        )

        del X_train_1, X_valid_1, y_train_1, y_valid_1
        pred_model = catclf
        del catclf

    elif model_type == "randomforest":
        rfclf = RandomForestClassifier(
            n_estimators=512, bootstrap=True, max_features="sqrt"
        )

        rfclf.fit(X_train_, y_train_)
        pred_model = rfclf
        del rfclf

    

    else:
        print("Please, try one of the possible models")

    del X_train_, y_train_
    print("finish train")

    return pred_model, X_valid.copy(), y_valid.copy()



