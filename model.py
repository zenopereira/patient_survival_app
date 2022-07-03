import joblib
import os
import xgboost
import pandas as pd

curr_path = os.path.dirname(os.path.abspath(__file__))

xgb_final = joblib.load(curr_path + "/xgb_tuned_final.pkl.compressed")

def survival_predictions(x_test):

    if type(x_test) == type(pd.DataFrame()):
        y_hat = xgb_final.predict(x_test._get_numeric_data())
    else:
        y_hat = xgb_final.predict(x_test)

    return y_hat