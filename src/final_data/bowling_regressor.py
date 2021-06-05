import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPClassifier
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

RF = RandomForestClassifier(n_estimators=100)
gnb = GaussianNB()
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 5), random_state=1, max_iter=5000)
SVM = svm.SVC(kernel='linear', C=1)
rfr = RandomForestRegressor(max_depth=100, random_state=0)
reg = LinearRegression()
mltreg = MultiOutputRegressor(rfr)
predictor = mltreg

dirname = os.path.dirname(__file__)
dataset_source = os.path.join(dirname, "output\\bowling_encoded.csv")

input_data = pd.read_csv(dataset_source)
model_output_columns = output_bowling_columns
training_input_columns = input_bowling_columns.copy()
training_input_columns.remove("player_name")
remove_columns = [
    # "bowling_consistency",
    # "bowling_form",
    "bowling_wind",
    "bowling_rain",
    "bowling_session",
    "bowling_viscosity",
    "toss",
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


def calculate_econ(row):
    if row["deliveries"] == 0:
        return 0
    return row["runs_conceded"] * 6 / row["deliveries"]


def predict_bowling(dataset):
    scaled_dataset = input_scaler.transform(dataset.drop(columns=remove_columns))
    predicted = predictor.predict(scaled_dataset)
    result = pd.DataFrame(output_scaler.inverse_transform(predicted), columns=output_bowling_columns)
    for column in y.columns:
        dataset[column] = result[column]
    dataset["econ"] = dataset.apply(lambda row: calculate_econ(row), axis=1)
    return dataset


def bowling_predict_test():
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
    bowling_predict_test()
