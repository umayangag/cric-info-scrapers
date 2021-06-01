import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn import svm
import numpy as np
from team_selection.dataset_definitions import *
from sklearn import preprocessing

# from sklearn import preprocessing

RF = RandomForestClassifier(n_estimators=100, criterion='entropy', bootstrap=False, max_depth=100,
                            class_weight={0: 4, 1: 1, 2: 2})
gnb = GaussianNB()
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(4, 3), random_state=1)
SVM = svm.SVC(kernel='linear', C=1)
regr = RandomForestRegressor(max_depth=4, random_state=0)
reg = LinearRegression()
mltreg = MultiOutputRegressor(regr)
predictor = RF

dirname = os.path.dirname(__file__)
dataset_source = os.path.join(dirname, "output\\batting_encoded.csv")

input_data = pd.read_csv(dataset_source)
training_input_columns = input_batting_columns.copy()
training_input_columns.remove("player_name")

# scaler = preprocessing.StandardScaler().fit(input_data)
# data_scaled = scaler.transform(input_data)
# final_df = pd.DataFrame(data=data_scaled, columns=input_data.columns)
# input_data = final_df

X = input_data[training_input_columns]
y = input_data["runs_scored"]  # Labels
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
train_set = 2095
X_train = X.iloc[:train_set, :]
X_test = X.iloc[train_set + 1:, :]
y_train = y.iloc[:train_set]
y_test = y.iloc[train_set + 1:]
# predictor.fit(X_train, y_train)


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
    # y_pred = predictor.predict(X_test)
    # print(X_test)
    # comparison = {}
    # comparison["actual"] = y_test.to_numpy()
    # comparison["predicted"] = y_pred
    #
    # for i in range(0, len(y_pred)):
    #     print(comparison["actual"][i], " ", comparison["predicted"][i], "")
    # print("Error:", metrics.mean_absolute_error(y_test, y_pred))
    print("Cross Validation Score:", cross_val_score(predictor, X, y, scoring='accuracy', cv=10).mean())


if __name__ == "__main__":
    batting_predict_test()
    # predict_batting("", 1, 1)
