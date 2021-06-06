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
import math
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor

rfr = RandomForestRegressor(bootstrap=True, min_impurity_decrease=0.0, min_weight_fraction_leaf=0.0, max_depth=100,
                            n_estimators=100, max_features='auto', random_state=0,
                            criterion='mse')
lr = LinearRegression()
mlpr = MLPRegressor(random_state=3, max_iter=2000, activation='tanh', solver='sgd', hidden_layer_sizes=(17))
gb = GradientBoostingRegressor(n_estimators=30, loss='ls', criterion='mse')
mltreg = MultiOutputRegressor(rfr)
predictor = gb

dirname = os.path.dirname(__file__)
dataset_source = os.path.join(dirname, "output\\batting_encoded.csv")

input_data = pd.read_csv(dataset_source)

# input_data_normalized = input_data / input_data.sum()
# input_data_normalized["season"] = input_data["season"]
# input_data = input_data_normalized
model_output_columns = ["runs_scored"]
training_input_columns = input_batting_columns.copy()
training_input_columns.remove("player_name")
remove_columns = [
    "batting_consistency",
    "batting_form",
    "batting_wind",
    "batting_rain",
    # "batting_session",
    # "season",
    "batting_viscosity",
    "batting_inning",
    "batting_pressure",
    "batting_temp",
    "batting_cloud",
    "batting_humidity",
    "toss",
]

season_index = 18
training_data = input_data.loc[input_data["season"] < season_index]
test_data = input_data.loc[input_data["season"] >= season_index]

X = training_data[training_input_columns].drop(columns=remove_columns)
y = training_data[model_output_columns]  # Labels
# y['runs_scored'] = y['runs_scored'].apply(lambda x:pow(x,3))
# input_scaler = preprocessing.StandardScaler().fit(X)
# output_scaler = preprocessing.StandardScaler().fit(y)

X_train = pd.DataFrame(data=training_data[X.columns], columns=X.columns)
y_train = pd.DataFrame(data=training_data[model_output_columns], columns=model_output_columns)
X_test = pd.DataFrame(data=test_data[X.columns], columns=X.columns)
y_test = pd.DataFrame(data=test_data[model_output_columns], columns=model_output_columns)

# construct the initial dataset for SmoteR
# cols = X_train.columns.tolist()
# cols.append('runs_scored')
# D = pd.DataFrame(np.concatenate([X_train, y_train], axis=1), columns=cols)
# Xs = SmoteR(D, target='runs_scored', th=0.8, o=100, u=50, k=4, categorical_col=[])

# X_train = Xs.drop(columns=['runs_scored'])
# y_train = Xs[['runs_scored']]

predictor.fit(X_train, y_train.values.ravel())


def optimizer():
    n = 1  # how many times to shuffle the training data
    nhn_range = [8, 10, 12, 14, 16, 18]  # number of hidden neurons

    score_dict = {}
    for nhn in nhn_range:
        mlp = MLPRegressor(hidden_layer_sizes=(nhn,), activation='tanh',
                           solver='sgd', shuffle=False, random_state=42,
                           max_iter=20000, momentum=0.7, early_stopping=True,
                           validation_fraction=0.15)

        nhn_scores = []
        for _ in range(n):
            df_train = shuffle(X_train)
            score = np.sqrt(-cross_val_score(mlp, df_train[X_train.columns],
                                             y_train["runs_scored"],
                                             cv=10, scoring='neg_mean_squared_error')).mean()
            nhn_scores.append(score)
        score_dict[nhn] = nhn_scores
        score_df = pd.DataFrame.from_dict(score_dict)
        score_df.to_csv("optimize_batting.csv")


def calculate_strike_rate(row):
    if row["balls_faced"] == 0:
        return 0
    return row["runs_scored"] * 100 / row["balls_faced"]


def predict_batting(dataset):
    scaled_dataset = dataset
    predicted = predictor.predict(scaled_dataset)
    result = pd.DataFrame(predicted, columns=output_batting_columns)
    print(result)
    for column in y.columns:
        dataset[column] = result[column]
    dataset["strike_rate"] = dataset.apply(lambda row: calculate_strike_rate(row), axis=1)
    return dataset


def batting_predict_test():
    y_pred = predictor.predict(X_test)
    # print(X_test)

    plt.figure(figsize=(6 * 1.618, 6))
    index = np.arange(len(X.columns))
    bar_width = 0.35
    plt.barh(index, predictor.feature_importances_, color='black', alpha=0.5)
    plt.ylabel('features')
    plt.xlabel('importance')
    plt.title('Feature importance')
    plt.yticks(index, X.columns)
    plt.tight_layout()
    plt.show()

    plt.plot(range(0, len(y_test)), y_test, color='blue')
    plt.plot(range(0, len(y_pred)), y_pred, color='red')
    plt.title('Actual vs Predicted')
    plt.xlabel('Instance')
    plt.ylabel('Runs Scored')
    plt.show()

    plt.scatter(y_train, predictor.predict(X_train), color='red')
    plt.plot(y_train, y_train, color='blue')
    plt.scatter(y_test, predictor.predict(X_test), color='green')
    plt.title('Actual vs Predicted')
    plt.xlabel('Actual Runs Scored')
    plt.ylabel('Predicted Runs Scored')
    plt.show()

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('R2:', metrics.r2_score(y_test, y_pred))
    print('R2:', metrics.r2_score(y_train, predictor.predict(X_train)))
    #
    # print("Cross Validation Score:", cross_val_score(predictor, X, y, cv=10).mean())


if __name__ == "__main__":
    batting_predict_test()
    # optimizer()
    # predict_batting("", 1, 1)
