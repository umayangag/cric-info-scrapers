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
from final_data.encoders import *

RF = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=1000,
                            class_weight={0:1, 1: 1, 2: 1})
gnb = GaussianNB()
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 5), random_state=1, max_iter=5000)
SVM = svm.SVC(kernel='linear', C=1)
regr = RandomForestRegressor(max_depth=2, random_state=0)
reg = LinearRegression()
mltreg = MultiOutputRegressor(reg)
predictor = RF

dirname = os.path.dirname(__file__)
dataset_source = os.path.join(dirname, "output\\bowling_encoded.csv")

input_data = pd.read_csv(dataset_source)
training_input_columns = input_bowling_columns.copy()
training_input_columns.remove("player_name")

X = input_data[training_input_columns]
y = input_data["wickets_taken"]  # Labels
y = y.apply(encode_wickets)
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

# scaler = preprocessing.StandardScaler().fit(X)
# data_scaled = scaler.transform(X)
# final_df = pd.DataFrame(data=data_scaled, columns=training_input_columns)
# X = final_df

train_set = 1465
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# X_train = X.iloc[:train_set, :]
# X_test = X.iloc[train_set + 1:, :]
# y_train = y.iloc[:train_set]
# y_test = y.iloc[train_set + 1:]
predictor.fit(X_train, y_train)

predictor.fit(X_train, y_train)


def calculate_econ(row):
    if row["deliveries"] == 0:
        return 0
    return row["runs_conceded"] * 6 / row["deliveries"]


def predict_bowling(dataset):
    predicted = predictor.predict(dataset)
    result = pd.DataFrame(predicted, columns=y.columns)
    for column in y.columns:
        dataset[column] = result[column]
    dataset["econ"] = dataset.apply(lambda row: calculate_econ(row), axis=1)
    return dataset


def bowling_predict_test():
    y_pred = predictor.predict(X_test)
    #
    # comparison = {}
    # comparison["actual"] = y_test.to_numpy()
    # comparison["predicted"] = y_pred
    #
    # for i in range(0, len(y_pred)):
    #     print(comparison["actual"][i], " ", comparison["predicted"][i], "", )
    #
    # print("Error:", metrics.mean_absolute_error(y_test, y_pred))

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

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred, labels=[0, 1, 2]))

    # print("Cross Validation Score:", cross_val_score(predictor, X, y, cv=10).max())


if __name__ == "__main__":
    bowling_predict_test()
