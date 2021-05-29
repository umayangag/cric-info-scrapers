import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn import svm

RF = RandomForestClassifier(n_estimators=100, criterion='entropy', bootstrap=False, max_depth=100,
                            class_weight={0: 1, 1: 1})
gnb = GaussianNB()
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 6), random_state=1, max_iter=10000)
SVM = svm.SVC(kernel='linear', C=1)

dirname = os.path.dirname(__file__)
dataset_source = os.path.join(dirname, "output\\batting_win_predict_encoded.csv")


def batting_predict(predictor):
    input_data = pd.read_csv(dataset_source)
    X = input_data[[
        "contribution",
        "total_score",
        "total_balls",
        "batting_position",
        "player_consistency",
        "player_form",
        "runs",
        "balls",
        "fours",
        "sixes",
        "strike_rate",
        "temp",
        "wind",
        "rain",
        "humidity",
        "cloud",
        "pressure",
        "viscosity",
        "inning",
        "batting_session",
        "toss",
        "venue",
        "opposition",
        "season",
    ]]  # Features
    y = input_data["result"]  # Labels

    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)
    print(X)
    train_set = 2000

    X_train = X.iloc[:train_set, :]
    X_test = X.iloc[train_set + 1:, :]
    y_train = y.iloc[:train_set]
    y_test = y.iloc[train_set + 1:]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    #
    predictor.fit(X_train, y_train)
    y_pred = predictor.predict(X_test)
    probability = predictor.predict_proba(X_test)
    # print(type(y_pred), type(y_test))
    # print(type(probability))
    # for i, row in y_test.items():
    #     index=i-(train_set+1)
    #     print(i, row, y_pred[index], probability[index])

    # for classifiers
    # print("Score:", predictor.score(X_test, y_test))
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    # print("Cross Validation Score:", cross_val_score(predictor, X, y, cv=5).min())
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    # print(predictor.get_params())
    return accuracy


batting_predict(clf)
# values = []
# for i in range(1, 20):
#     for j in range(1, 20):
#         cf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(i, j), random_state=1)
#         value = batting_predict(cf)
#         values.append([i, j, value])
# print(values)

# i = 0
# j = 0
# max = 0
# for item in array:
#     if item[2] > max:
#         i = item[0]
#         j = item[1]
#         max = item[2]
# print(i,j,max)
