import os
import pickle
from analyze.error_curves import get_error_curves

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

from team_selection.dataset_definitions import *

model_file = "batting_performance_predictor.sav"
corrector_file = "batting_performance_corrector.sav"
scaler_file = "batting_scaler.sav"
offset_array = [3.3652, 3.6601, 0.3073, 0.1321, -0.5213]
input_columns = [
    'batting_consistency',
    'batting_form',
    'batting_temp',
    'batting_wind',
    'batting_rain',
    'batting_humidity',
    'batting_cloud',
    'batting_pressure',
    'batting_viscosity',
    'batting_inning',
    'batting_session',
    'toss',
    'venue',
    'opposition',
    'season',
]


def predict_batting(dataset):
    loaded_predictor = pickle.load(open(model_file, 'rb'))
    loaded_corrector = pickle.load(open(corrector_file, 'rb'))
    loaded_scaler = pickle.load(open(scaler_file, 'rb'))
    predicted = loaded_predictor.predict(loaded_scaler.transform(dataset[input_columns]))
    predicted_df = pd.DataFrame(predicted, columns=output_batting_columns)
    corrections = loaded_corrector.predict(predicted_df)
    corrections_df = pd.DataFrame(corrections, columns=output_batting_columns)
    result = predicted_df - corrections_df
    result[result < 0] = 0
    for column in output_batting_columns:
        dataset[column] = result[column]
    dataset["strike_rate"] = dataset.apply(lambda row: calculate_strike_rate(row), axis=1)
    return dataset


def calculate_strike_rate(row):
    if row["balls_faced"] == 0:
        return 0
    return row["runs_scored"] * 100 / row["balls_faced"]


def batting_predict_test():
    y_pred = predictor.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=output_batting_columns)

    plt.figure(figsize=(6 * 1.618, 6))
    index = np.arange(len(X.columns))
    plt.barh(index, predictor.feature_importances_, color='black', alpha=0.5)
    plt.ylabel('features')
    plt.xlabel('importance')
    plt.title('Feature importance')
    plt.yticks(index, X.columns)
    plt.tight_layout()
    plt.show()

    # plt.plot(range(0, len(y_test)), y_test["runs_scored"], color='red')
    # plt.plot(range(0, len(y_pred)), y_pred["runs_scored"], color='blue')
    # plt.title('Actual vs Predicted')
    # plt.xlabel('Instance')
    # plt.ylabel('Runs Scored')
    # plt.show()

    # predict runs
    plt.scatter(y_train["runs_scored"], train_predict["runs_scored"], color='red', s=2)
    plt.plot(y_train["runs_scored"], y_train["runs_scored"], color='blue')
    plt.scatter(y_test["runs_scored"], y_pred["runs_scored"], color='green', s=4)
    plt.title('Actual vs Predicted')
    plt.xlabel('Actual Runs Scored')
    plt.ylabel('Predicted Runs Scored')
    plt.show()

    # # train error
    # plt.scatter(y_train["runs_scored"], train_predict["runs_scored"] - y_train["runs_scored"], color='red', s=2)
    # plt.plot(y_train["runs_scored"], y_train["runs_scored"] - y_train["runs_scored"], color='blue')
    # plt.scatter(y_test["runs_scored"], y_pred["runs_scored"] - y_test.reset_index()["runs_scored"], color='green', s=4)
    # plt.title('Actual vs Predicted Residuals')
    # plt.xlabel('Actual Runs Scored')
    # plt.ylabel('Predicted Runs Scored Residuals')
    # plt.show()

    train_correct = pd.DataFrame(corrector.predict(y_train), columns=output_batting_columns) - offset_array
    test_correct = pd.DataFrame(corrector.predict(y_test), columns=output_batting_columns) - offset_array

    # # predict error
    # plt.scatter(y_train["runs_scored"], train_predict["runs_scored"] - y_train["runs_scored"], color='red', s=2)
    # plt.scatter(y_train["runs_scored"], train_correct["runs_scored"], color='blue', s=2)
    # plt.scatter(y_test["runs_scored"], test_correct["runs_scored"], color='green', s=2)
    # plt.title('Actual vs Predicted Residuals')
    # plt.xlabel('Actual Runs Scored')
    # plt.ylabel('Predicted Runs Scored')
    # plt.show()

    for attribute in output_batting_columns:
        # corrected runs
        plt.scatter(y_train[attribute], train_predict[attribute] - train_correct[attribute], color='red',
                    s=2)
        plt.plot(y_train[attribute], y_train[attribute], color='blue')
        plt.scatter(y_test[attribute], y_pred[attribute] - test_correct[attribute], color='green', s=4)
        plt.title('Actual vs Predicted')
        plt.xlabel('Actual ' + attribute)
        plt.ylabel('Predicted ' + attribute)
        plt.show()

        # print(attribute)
        # print("Training Set")
        # print('Mean Absolute Error:',
        #       metrics.mean_absolute_error(y_train[attribute], train_predict[attribute] - train_correct[attribute]))
        # print('Mean Squared Error:',
        #       metrics.mean_squared_error(y_train[attribute], train_predict[attribute] - train_correct[attribute]))
        # print('Root Mean Squared Error:',
        #       np.sqrt(
        #           metrics.mean_squared_error(y_train[attribute], train_predict[attribute] - train_correct[attribute])))
        # print('R2:', metrics.r2_score(y_train[attribute], train_predict[attribute] - train_correct[attribute]))
        # print("-----------------------------------------------------------------------------------")
        # print("Test Set")
        # print('Mean Squared Error:',
        #       metrics.mean_squared_error(y_test[attribute], y_pred[attribute] - test_correct[attribute]))
        # print('Root Mean Squared Error:',
        #       np.sqrt(metrics.mean_squared_error(y_test[attribute], y_pred[attribute] - test_correct[attribute])))

        # max_r2 = 0
        # chosen_i = 0
        # start = -2
        # L = 0.0001
        # for i in range(20000):
        #     r2 = metrics.r2_score(y_test[attribute], y_pred[attribute] - test_correct[attribute] + start + L * i)
        #     if r2 > max_r2:
        #         max_r2 = r2
        #         chosen_i = start + L * i
        # print('max R2:', chosen_i, ":", max_r2)

        print(attribute,'R2:', metrics.r2_score(y_test[attribute], y_pred[attribute] - test_correct[attribute]))
        print("-----------------------------------------------------------------------------------")
        # exit()


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    dataset_source = os.path.join(dirname, "output\\batting_encoded.csv")

    input_data = pd.read_csv(dataset_source)
    # input_data = input_data.sample(frac=1).reset_index(drop=True)
    training_input_columns = input_batting_columns.copy()
    training_input_columns.remove("player_name")

    X = input_data[input_columns]
    y = input_data[output_batting_columns].copy()  # Labels

    # scaler = preprocessing.StandardScaler().fit(X)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
    pickle.dump(scaler, open(scaler_file, 'wb'))
    data_scaled = scaler.transform(X)
    final_df = pd.DataFrame(data=data_scaled, columns=input_columns)
    X = final_df
    X.to_csv("final_batting_2021.csv")

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    train_set = 2106
    train_set = 1794
    X_train = X.iloc[:train_set, :]
    X_test = X.iloc[train_set + 1:, :]
    y_train = y.iloc[:train_set]
    y_test = y.iloc[train_set + 1:]

    predictor = RandomForestRegressor(max_depth=6, n_estimators=200, random_state=1, max_features="auto",
                                      n_jobs=-1)
    predictor.fit(X_train, y_train)

    train_predict = pd.DataFrame(predictor.predict(X_train), columns=output_batting_columns)

    corrector = RandomForestRegressor(max_depth=6, n_estimators=100, random_state=1, max_features="auto",
                                      n_jobs=-1)
    corrector.fit(y_train, train_predict - y_train)

    # get_error_curves(X_train, y_train, X_test, y_test, output_batting_columns, 10, "runs_scored")
    pickle.dump(predictor, open(model_file, 'wb'))
    pickle.dump(corrector, open(corrector_file, 'wb'))
    batting_predict_test()
