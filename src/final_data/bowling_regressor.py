import os
import pickle
from analyze.error_curves import get_error_curves
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

from team_selection.dataset_definitions import *

model_file = "bowling_performance_predictor.sav"
corrector_file = "bowling_performance_corrector.sav"
wicket_corrector_file = "wicket_corrector.sav"
offset_array = [-1.591, -0.1940, 0.0248]
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
    loaded_corrector = pickle.load(open(corrector_file, 'rb'))
    loaded_wicket_corrector = pickle.load(open(wicket_corrector_file, 'rb'))
    loaded_scaler = pickle.load(open(scaler_file, 'rb'))
    predicted = loaded_predictor.predict(loaded_scaler.transform(dataset[input_columns]))
    predicted_df = pd.DataFrame(predicted, columns=output_bowling_columns)
    corrections = loaded_corrector.predict(predicted_df)
    corrected_wickets = loaded_wicket_corrector.predict(predicted_df)
    corrections_df = pd.DataFrame(corrections, columns=output_bowling_columns)
    result = predicted_df - corrections_df + offset_array
    result["wickets_taken"] = result["wickets_taken"] - corrected_wickets
    result[result < 0] = 0
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

    # plt.plot(range(0, len(y_test)), y_test["runs_conceded"], color='red')
    # plt.plot(range(0, len(y_pred)), y_pred["runs_conceded"], color='blue')
    # plt.title('Actual vs Predicted')
    # plt.xlabel('Instance')
    # plt.ylabel('Runs Scored')
    # plt.show()
    #
    # plt.scatter(y_train["runs_conceded"], train_predict["runs_conceded"].apply(lambda x: x * 3 - 75), color='red')
    # plt.plot(y_train["runs_conceded"], y_train["runs_conceded"], color='blue')
    # plt.scatter(y_test["runs_conceded"], y_pred["runs_conceded"].apply(lambda x: x * 3 - 75), color='green')
    # plt.title('Actual vs Predicted')
    # plt.xlabel('Actual Runs Conceded')
    # plt.ylabel('Predicted Runs Conceded')
    # plt.show()
    #
    # # train error
    # plt.scatter(y_train["runs_conceded"], train_predict["runs_conceded"] - y_train["runs_conceded"], color='red', s=2)
    # plt.plot(y_train["runs_conceded"], y_train["runs_conceded"] - y_train["runs_conceded"], color='blue')
    # plt.scatter(y_test["runs_conceded"], y_pred["runs_conceded"] - y_test.reset_index()["runs_conceded"], color='green',
    #             s=4)
    # plt.title('Actual vs Predicted Residuals')
    # plt.xlabel('Actual Runs Scored')
    # plt.ylabel('Predicted Runs Scored Residuals')
    # plt.show()
    #
    # corrector = RandomForestRegressor(max_depth=100, n_estimators=200, random_state=1, max_features="auto",
    #                                   n_jobs=-1)
    # corrector.fit(y_train, train_predict - y_train)
    train_correct = pd.DataFrame(corrector.predict(y_train), columns=output_bowling_columns) - offset_array
    train_wicket_correct = pd.DataFrame(wicket_corrector.predict(y_train), columns=["wickets_taken"])
    test_correct = pd.DataFrame(corrector.predict(y_test), columns=output_bowling_columns) - offset_array
    test_wicket_correct = pd.DataFrame(wicket_corrector.predict(y_test), columns=["wickets_taken"])
    #
    # # predict error
    # plt.scatter(y_train["runs_conceded"], train_predict["runs_conceded"] - y_train["runs_conceded"], color='red', s=2)
    # plt.scatter(y_train["runs_conceded"], train_correct["runs_conceded"], color='blue', s=2)
    # plt.scatter(y_test["runs_conceded"], test_correct["runs_conceded"], color='green', s=2)
    # plt.title('Actual vs Predicted Residuals')
    # plt.xlabel('Actual Runs Scored')
    # plt.ylabel('Predicted Runs Scored')
    # plt.show()
    #
    # # corrected runs
    # plt.scatter(y_train["runs_conceded"], train_predict["runs_conceded"] - train_correct["runs_conceded"], color='red',
    #             s=2)
    # plt.plot(y_train["runs_conceded"], y_train["runs_conceded"], color='blue')
    # plt.scatter(y_test["runs_conceded"], y_pred["runs_conceded"] - test_correct["runs_conceded"], color='green', s=4)
    # plt.title('Actual vs Predicted')
    # plt.xlabel('Actual Runs Scored')
    # plt.ylabel('Predicted Runs Scored')
    # plt.show()

    for attribute in output_bowling_columns:
        if attribute != "wickets_taken":
            plt.scatter(y_train[attribute], train_predict[attribute] - train_correct[attribute], color='red',
                        s=2, label="training data")
            plt.plot(y_train[attribute], y_train[attribute], color='blue')
            plt.scatter(y_test[attribute], y_pred[attribute] - test_correct[attribute], color='green', s=4,label="test data")
            plt.title('Actual vs Predicted')
            plt.xlabel('Actual ' + attribute)
            plt.ylabel('Predicted ' + attribute)
            plt.legend()
            plt.show()
            print('Root Mean Squared Error:',
                  np.sqrt(metrics.mean_squared_error(y_test[attribute], y_pred[attribute] - test_correct[attribute])))
            print(attribute, 'R2:', metrics.r2_score(y_test[attribute], y_pred[attribute] - test_correct[attribute]))
        else:
            plt.scatter(y_train[attribute], train_predict[attribute] - train_wicket_correct[attribute], color='red',
                        s=2, label="training data")
            plt.plot(y_train[attribute], y_train[attribute], color='blue')
            plt.scatter(y_test[attribute], y_pred[attribute] - test_wicket_correct[attribute], color='green', s=4,label="test data")
            plt.title('Actual vs Predicted')
            plt.xlabel('Actual ' + attribute)
            plt.ylabel('Predicted ' + attribute)
            plt.legend()
            plt.show()
            print('Root Mean Squared Error:',
                  np.sqrt(metrics.mean_squared_error(y_test[attribute], y_pred[attribute] - test_correct[attribute])))
            print(attribute, 'R2:',
                  metrics.r2_score(y_test[attribute], y_pred[attribute] - test_wicket_correct[attribute]))

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
        # start = -1
        # L =0.0001
        # for i in range(20000):
        #     r2 = metrics.r2_score(y_test[attribute], y_pred[attribute] - test_correct[attribute] + start + L * i)
        #     if r2 > max_r2:
        #         max_r2 = r2
        #         chosen_i = start + L * i
        # print('max R2:', chosen_i, ":", max_r2)

        print("-----------------------------------------------------------------------------------")
        # exit()


if __name__ == "__main__":
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

    RFR = RandomForestRegressor(max_depth=6, n_estimators=200, random_state=0, n_jobs=-1)
    predictor = RFR

    predictor.fit(X_train, y_train)

    train_predict = pd.DataFrame(predictor.predict(X_train), columns=output_bowling_columns)

    corrector = RandomForestRegressor(max_depth=6, n_estimators=500, random_state=1, max_features="auto",
                                      n_jobs=-1)
    wicket_corrector = RandomForestRegressor(max_depth=6, n_estimators=500, random_state=1, max_features="auto",
                                             n_jobs=-1)
    corrector.fit(y_train, train_predict - y_train)
    wicket_corrector.fit(y_train, train_predict["wickets_taken"] - y_train["wickets_taken"])

    # get_error_curves(X_train, y_train, X_test, y_test, output_bowling_columns, 25, "runs_conceded")
    pickle.dump(predictor, open(model_file, 'wb'))
    pickle.dump(corrector, open(corrector_file, 'wb'))
    pickle.dump(wicket_corrector, open(wicket_corrector_file, 'wb'))
    bowling_predict_test()
    # predict_bowling("", 1, 1)
