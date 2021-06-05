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
training_input_columns = input_batting_columns.copy()
training_input_columns.remove("player_name")
# training_input_columns.remove("batting_consistency")
# training_input_columns.remove("batting_form")
# training_input_columns.remove("batting_wind")
# training_input_columns.remove("batting_rain")
# training_input_columns.remove("batting_session")
# training_input_columns.remove("season")

# construct the initial dataset for SmoteR
input_data = SmoteR(input_data, target='runs_scored', th=0.6, o=2000, u=80, k=4, categorical_col=[])
X = input_data[training_input_columns]
y = input_data[output_batting_columns]  # Labels

input_scaler = preprocessing.StandardScaler().fit(X)
# input_data_scaled = input_scaler.transform(X)
# X = pd.DataFrame(data=input_data_scaled, columns=X.columns)

output_scaler = preprocessing.StandardScaler().fit(y)
# output_data_scaled = output_scaler.transform(y)
# y = pd.DataFrame(data=output_data_scaled, columns=output_batting_columns)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
season_index = 20
X_train = input_data.loc[input_data['season'] < season_index][training_input_columns]
X_test = input_data.loc[input_data['season'] >= season_index][training_input_columns]
y_train = input_data.loc[input_data['season'] < season_index][output_batting_columns]
y_test = input_data.loc[input_data['season'] >= season_index][output_batting_columns]

predictor.fit(X_train, y_train)


def calculate_strike_rate(row):
    if row["balls_faced"] == 0:
        return 0
    return row["runs_scored"] * 100 / row["balls_faced"]


def predict_batting(dataset):
    scaled_dataset = input_scaler.transform(dataset)
    predicted = predictor.predict(scaled_dataset)
    result = pd.DataFrame(output_scaler.inverse_transform(predicted), columns=output_batting_columns)
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

    plt.plot(range(0, len(y_test)), y_test, color='red')
    plt.plot(range(0, len(y_pred)), y_pred, color='blue')
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
    #
    # print("Cross Validation Score:", cross_val_score(predictor, X, y, cv=10).mean())


if __name__ == "__main__":
    batting_predict_test()
    # predict_batting("", 1, 1)
