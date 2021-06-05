import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from team_selection.dataset_definitions import *
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from final_data.smoter import SmoteR

rfr = RandomForestRegressor(bootstrap=True, min_impurity_decrease=0.0, min_weight_fraction_leaf=0.0, max_depth=100,
                            n_estimators=100, max_features='auto', random_state=0,
                            criterion='mse')
lr = LinearRegression()
mlpr = MLPRegressor(random_state=1, max_iter=2000)
mltreg = MultiOutputRegressor(rfr)
predictor = mltreg

dirname = os.path.dirname(__file__)
dataset_source = os.path.join(dirname, "output\\batting_encoded.csv")

input_data = pd.read_csv(dataset_source)
model_output_columns= output_batting_columns
training_input_columns = input_batting_columns.copy()
training_input_columns.remove("player_name")
remove_columns = [
    # "batting_consistency",
    # "batting_form",
    # "batting_wind",
    # "batting_rain",
    # "batting_session",
    # "season"
]

season_index = 20
training_data = input_data.loc[input_data["season"] < season_index]
test_data = input_data.loc[input_data["season"] >= season_index]

X = training_data[training_input_columns].drop(columns=remove_columns)
y = training_data[model_output_columns]  # Labels

input_scaler = preprocessing.StandardScaler().fit(X)
output_scaler = preprocessing.StandardScaler().fit(y)

X_train = pd.DataFrame(data=input_scaler.transform(training_data[X.columns]), columns=X.columns)
y_train = pd.DataFrame(data=output_scaler.transform(training_data[model_output_columns]), columns=model_output_columns)
X_test = pd.DataFrame(data=input_scaler.transform(test_data[X.columns]), columns=X.columns)
y_test = pd.DataFrame(data=output_scaler.transform(test_data[model_output_columns]), columns=model_output_columns)

predictor.fit(X_train, y_train)


def calculate_strike_rate(row):
    if row["balls_faced"] == 0:
        return 0
    return row["runs_scored"] * 100 / row["balls_faced"]


def predict_batting(dataset):
    scaled_dataset = input_scaler.transform(dataset.drop(columns=remove_columns))
    print(X_train.columns)
    print(dataset.columns)
    predicted = predictor.predict(scaled_dataset)
    result = pd.DataFrame(output_scaler.inverse_transform(predicted), columns=output_batting_columns)
    for column in y.columns:
        dataset[column] = result[column]
    dataset["strike_rate"] = dataset.apply(lambda row: calculate_strike_rate(row), axis=1)
    return dataset


def batting_predict_test():
    y_pred = predictor.predict(X_test)
    # print(X_test)

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('R2:', metrics.r2_score(y_test, y_pred))
    #
    # print("Cross Validation Score:", cross_val_score(predictor, X, y, cv=10).mean())


if __name__ == "__main__":
    batting_predict_test()
    # predict_batting("", 1, 1)
