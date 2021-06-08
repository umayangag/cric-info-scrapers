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

RF = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=1000)
predictor = RF

dirname = os.path.dirname(__file__)
dataset_source = os.path.join(dirname, "output\\batting_encoded.csv")

input_data = pd.read_csv(dataset_source)
input_data = input_data.sample(frac=1).reset_index(drop=True)
training_input_columns = input_batting_columns.copy()
training_input_columns.remove("player_name")
input_columns=[
    'batting_consistency',
    'batting_form',
    'batting_temp',
    'batting_wind',
    # 'batting_rain',
    'batting_humidity',
    'batting_cloud',
    'batting_pressure',
    # 'batting_viscosity',
    # 'batting_inning',
    # 'batting_session',
    # 'toss',
    'venue',
    'opposition',
    'season',
]
X = input_data[input_columns]
y = input_data[output_batting_columns]  # Labels
y["runs_scored"] = y["runs_scored"].apply(encode_runs)
y["balls_faced"] = y["balls_faced"].apply(encode_balls_faced)
y["fours_scored"] = y["fours_scored"].apply(encode_fours)
y["sixes_scored"] = y["sixes_scored"].apply(encode_sixes)
y["batting_position"] = y["batting_position"].apply(encode_batting_position)


tempX = X
tempX["balls_faced"] = y["balls_faced"]
tempX["fours_scored"] = y["fours_scored"]
tempX["sixes_scored"] = y["sixes_scored"]
tempX["batting_position"] = y["batting_position"]

oversample = SMOTE()
tempX, runs_predicted = oversample.fit_resample(tempX, y["runs_scored"])
tempX["runs_scored"] = runs_predicted
X = tempX[input_columns]
y = tempX[output_batting_columns]

scaler = preprocessing.StandardScaler().fit(X)
data_scaled = scaler.transform(X)
final_df = pd.DataFrame(data=data_scaled, columns=input_columns)
X = final_df
# X.to_csv("final_batting_2021.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# train_set = 5000
# X_train = X.iloc[:train_set, :]
# X_test = X.iloc[train_set + 1:, :]
# y_train = y.iloc[:train_set]
# y_test = y.iloc[train_set + 1:]
predictor.fit(X_train, y_train)


def calculate_strike_rate(row):
    if row["balls_faced"] == 0:
        return 0
    return row["runs_scored"] * 100 / row["balls_faced"]


def predict_batting(dataset):
    predicted = predictor.predict(dataset)
    result = pd.DataFrame(predicted, columns=y.columns)
    for column in y.columns:
        dataset[column] = result[column]
    dataset["strike_rate"] = dataset.apply(lambda row: calculate_strike_rate(row), axis=1)
    return dataset


def batting_predict_test():
    y_pred = predictor.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=output_batting_columns)

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

    for attribute in output_batting_columns:
        print("Accuracy:" + attribute, metrics.accuracy_score(y_test[attribute], y_pred[attribute]))
        print(attribute + '\n', confusion_matrix(y_test[attribute], y_pred[attribute], labels=[0, 1, 2, 3, 4]))

    print("Cross Validation Score:", cross_val_score(predictor, X, y, scoring='accuracy', cv=10).mean())


if __name__ == "__main__":
    batting_predict_test()
    # predict_batting("", 1, 1)
