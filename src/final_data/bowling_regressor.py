import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
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
from sklearn.ensemble import GradientBoostingRegressor
import pickle

model_file = "bowling_performance_predictor.sav"
scaler_file = "bowling_scaler.sav"

input_columns = [
    "bowling_consistency",
    "bowling_form",
    "bowling_temp",
    "bowling_wind",
    "bowling_rain",
    "bowling_humidity",
    "bowling_cloud",
    "bowling_pressure",
    "bowling_viscosity",
    "batting_inning",
    "bowling_session",
    "toss",
    "bowling_venue",
    "bowling_opposition",
    "season",
]


def calculate_economy(row):
    if row["deliveries"] == 0:
        return 0
    return row["runs_conceded"] * 6 / row["deliveries"]


def predict_bowling(dataset):
    loaded_predictor = pickle.load(open(model_file, 'rb'))
    loaded_scaler = pickle.load(open(scaler_file, 'rb'))
    predicted = loaded_predictor.predict(loaded_scaler.transform(dataset[input_columns]))
    result = pd.DataFrame(predicted, columns=output_bowling_columns)
    for column in output_bowling_columns:
        dataset[column] = result[column]
    dataset["economy"] = dataset.apply(lambda row: calculate_economy(row), axis=1)
    return dataset


def bowling_predict_test():
    y_pred = predictor.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=output_bowling_columns)

    plt.figure(figsize=(6 * 1.618, 6))
    index = np.arange(len(X.columns))
    plt.barh(index, predictor.feature_importances_, color='black', alpha=0.5)
    plt.ylabel('features')
    plt.xlabel('importance')
    plt.title('Feature importance')
    plt.yticks(index, X.columns)
    plt.tight_layout()
    plt.show()

    train_predict = pd.DataFrame(predictor.predict(X_train), columns=output_bowling_columns)

    plt.plot(range(0, len(y_test)), y_test["runs_conceded"], color='red')
    plt.plot(range(0, len(y_pred)), y_pred["runs_conceded"], color='blue')
    plt.title('Actual vs Predicted')
    plt.xlabel('Instance')
    plt.ylabel('Runs Scored')
    plt.show()

    plt.scatter(y_train["runs_conceded"], train_predict["runs_conceded"], color='red')
    plt.plot(y_train["runs_conceded"], y_train["runs_conceded"], color='blue')
    plt.scatter(y_test["runs_conceded"], y_pred["runs_conceded"], color='green')
    plt.title('Actual vs Predicted')
    plt.xlabel('Actual Runs Conceded')
    plt.ylabel('Predicted Runs Conceded')
    plt.show()

    for attribute in output_bowling_columns:
        print(attribute)
        print('Mean Absolute Error:', metrics.mean_absolute_error(y_test[attribute], y_pred[attribute]))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test[attribute], y_pred[attribute]))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test[attribute], y_pred[attribute])))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_train[attribute], train_predict[attribute])))
        print('R2:', metrics.r2_score(y_test[attribute], y_pred[attribute]))
        print('R2:', metrics.r2_score(y_train[attribute], train_predict[attribute]))
        print("-----------------------------------------------------------------------------------")


if __name__ == "__main__":
    RFR = RandomForestRegressor(max_depth=1000, n_estimators=1000, random_state=0, n_jobs=-1)
    predictor = RFR

    dirname = os.path.dirname(__file__)
    dataset_source = os.path.join(dirname, "output\\bowling_encoded.csv")

    input_data = pd.read_csv(dataset_source)
    # input_data = input_data.sample(frac=1).reset_index(drop=True)
    training_input_columns = input_bowling_columns.copy()
    training_input_columns.remove("player_name")

    X = input_data[input_columns]
    y = input_data[output_bowling_columns].copy()  # Labels

    scaler = preprocessing.StandardScaler().fit(X)
    pickle.dump(scaler, open(scaler_file, 'wb'))
    data_scaled = scaler.transform(X)
    final_df = pd.DataFrame(data=data_scaled, columns=input_columns)
    X = final_df
    X.to_csv("final_bowling_2021.csv")

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    train_set = 1465
    train_set = 1268
    X_train = X.iloc[:train_set, :]
    X_test = X.iloc[train_set + 1:, :]
    y_train = y.iloc[:train_set]
    y_test = y.iloc[train_set + 1:]
    predictor.fit(X_train, y_train)
    pickle.dump(predictor, open(model_file, 'wb'))
    bowling_predict_test()
    # predict_bowling("", 1, 1)
