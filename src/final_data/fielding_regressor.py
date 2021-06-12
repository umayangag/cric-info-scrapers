import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDRegressor
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn import svm
import numpy as np
from team_selection.dataset_definitions import *
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from final_data.encoders import *
import pickle
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from analyze.error_curves import get_error_curves

model_file = "fielding_performance_predictor.sav"
scaler_file = "fielding_scaler.sav"

input_columns = [
    "fielding_consistency",
    "fielding_temp",
    "fielding_wind",
    # "fielding_rain",
    "fielding_humidity",
    "fielding_cloud",
    "fielding_pressure",
    # "fielding_viscosity",
    # "fielding_inning",
    # "fielding_session",
    # "toss",
    "season",
]


def predict_fielding(dataset):
    loaded_predictor = pickle.load(open(model_file, 'rb'))
    loaded_scaler = pickle.load(open(scaler_file, 'rb'))
    predicted = loaded_predictor.predict(loaded_scaler.transform(dataset[input_columns]))
    result = pd.DataFrame(predicted, columns=output_fielding_columns)
    for column in output_fielding_columns:
        dataset[column] = result[column]
    return dataset


def fielding_predict_test():
    y_pred = predictor.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=output_fielding_columns)

    plt.figure(figsize=(6 * 1.618, 6))
    index = np.arange(len(X.columns))
    plt.barh(index, predictor.feature_importances_, color='black', alpha=0.5)
    plt.ylabel('features')
    plt.xlabel('importance')
    plt.title('Feature importance')
    plt.yticks(index, X.columns)
    plt.tight_layout()
    plt.show()

    train_predict = pd.DataFrame(predictor.predict(X_train), columns=output_fielding_columns)

    plt.plot(range(0, len(y_test)), y_test["success_rate"], color='red')
    plt.plot(range(0, len(y_pred)), y_pred["success_rate"], color='blue')
    plt.title('Actual vs Predicted')
    plt.xlabel('Instance')
    plt.ylabel('success_rate')
    plt.show()

    gradient_corrected = train_predict["success_rate"]
    gradient_corrected = train_predict["success_rate"]

    plt.scatter(y_train["success_rate"], gradient_corrected, color='red', s=2)
    plt.plot(y_train["success_rate"], y_train["success_rate"], color='blue')
    plt.scatter(y_test["success_rate"], y_pred["success_rate"], color='green', s=4)
    plt.title('Actual vs Predicted')
    plt.xlabel('Actual success_rate')
    plt.ylabel('Predicted success_rate')
    plt.show()

    for attribute in output_fielding_columns:
        print(attribute)
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test[attribute], y_pred[attribute]))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test[attribute], y_pred[attribute]))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test[attribute], y_pred[attribute])))
        print('Root Mean Squared Error:',
              np.sqrt(metrics.mean_squared_error(y_train[attribute], train_predict[attribute])))
        print('R2:', metrics.r2_score(y_test[attribute], y_pred[attribute]))
        print('R2:', metrics.r2_score(y_train[attribute], train_predict[attribute]))
        print("-----------------------------------------------------------------------------------")
        exit()


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    dataset_source = os.path.join(dirname, "output\\fielding_encoded.csv")

    input_data = pd.read_csv(dataset_source)
    # input_data = input_data.sample(frac=1).reset_index(drop=True)
    training_input_columns = input_fielding_columns.copy()
    training_input_columns.remove("player_name")

    X = input_data[input_columns]
    y = input_data[output_fielding_columns].copy()  # Labels

    scaler = preprocessing.StandardScaler().fit(X)
    pickle.dump(scaler, open(scaler_file, 'wb'))
    data_scaled = scaler.transform(X)
    final_df = pd.DataFrame(data=data_scaled, columns=input_columns)
    X = final_df
    X.to_csv("final_fielding_2021.csv")

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    train_set = 873
    X_train = X.iloc[:train_set, :]
    X_test = X.iloc[train_set + 1:, :]
    y_train = y.iloc[:train_set]
    y_test = y.iloc[train_set + 1:]

    predictor = RandomForestRegressor(max_depth=1000, n_estimators=1000, random_state=1, max_features="auto",
                                      n_jobs=-1)
    predictor.fit(X_train, y_train["success_rate"])
    # get_error_curves(X_train, y_train, X_test, y_test, output_fielding_columns, 500)
    pickle.dump(predictor, open(model_file, 'wb'))
    fielding_predict_test()