import os
import pickle

import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np
from final_data.feature_select import select_features

model_file = "win_predictor.sav"
scaler_file = "win_predictor_scaler.sav"

all_columns = [
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
    'runs_scored',
    'balls_faced',
    'fours_scored',
    'sixes_scored',
    'batting_position',
    'batting_contribution',
    'strike_rate',
    'total_score',
    'total_wickets',
    'total_balls',
    'target',
    'extras',
    'match_number',
    'bowling_consistency',
    'bowling_form',
    'bowling_temp',
    'bowling_wind',
    'bowling_rain',
    'bowling_humidity',
    'bowling_cloud',
    'bowling_pressure',
    'bowling_viscosity',
    'bowling_session',
    'bowling_venue',
    'bowling_opposition',
    'runs_conceded',
    'deliveries',
    'wickets_taken',
    'bowling_contribution',
    "economy",
    "fielding_consistency",
    "success_rate"
]
all_columns = ['batting_temp', 'batting_wind', 'batting_rain', 'batting_humidity', 'batting_pressure',
               'batting_viscosity', 'batting_inning', 'runs_scored', 'strike_rate', 'total_score', 'total_wickets',
               'total_balls', 'target', 'extras', 'match_number', 'bowling_consistency', 'bowling_temp',
               'bowling_humidity', 'bowling_cloud', 'bowling_pressure', 'bowling_opposition', 'runs_conceded',
               'deliveries', 'wickets_taken', 'bowling_contribution']


def predict_for_team(input_team_data):
    team_data = input_team_data[all_columns].copy()
    loaded_predictor = pickle.load(open(model_file, 'rb'))
    loaded_scaler = pickle.load(open(scaler_file, 'rb'))
    team_performance = loaded_scaler.transform(team_data)
    predicted = loaded_predictor.predict_proba(team_performance)
    df = pd.DataFrame(predicted, columns=["lose", "win"])
    input_team_data["winning_probability"] = df["win"].to_numpy()
    return input_team_data.copy(), df["win"].mean()


def win_predict(predictor):
    y_pred = predictor.predict(X_test)

    r = permutation_importance(predictor, X_test, y_test, n_repeats=30, random_state=0)

    plt.figure(figsize=(6 * 1.618, 6))
    index = np.arange(len(X.columns))
    plt.barh(index, r.importances_mean, color='black', alpha=0.5)
    plt.ylabel('features')
    plt.xlabel('importance')
    plt.title('Feature importance')
    plt.yticks(index, X.columns)
    plt.tight_layout()
    plt.show()
    # probability = predictor.predict_proba(X_test)
    # print(type(y_pred), type(y_test))
    # print(type(probability))
    # for i, row in y_test.items():
    #     index=i-(train_set+1)
    #     print(i, row, y_pred[index], probability[index])

    # rfe = RFE(predictor, 10)
    # fit = rfe.fit(X, y)
    # print(all_columns)
    # print("Num Features: %d" % fit.n_features_)
    # print("Selected Features: %s" % fit.support_)
    # print("Feature Ranking: %s" % fit.ranking_)

    # for classifiers
    # print("Score:", predictor.score(X_test, y_test))

    confusion = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(confusion)

    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    print((TP + TN) / float(TP + TN + FP + FN))
    cvs = cross_val_score(predictor, X, y, scoring='accuracy', cv=10)
    print("Cross Validation Score:", cvs)
    print("Cross Validation Score:", cvs.mean())
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    classification_error = (FP + FN) / float(TP + TN + FP + FN)

    print("classification_error:", classification_error)

    sensitivity = TP / float(FN + TP)

    print("sensitivity:", sensitivity)

    specificity = TN / (TN + FP)

    print("specificity:", specificity)

    false_positive_rate = FP / float(TN + FP)

    print("false_positive_rate:", false_positive_rate)

    precision = TP / float(TP + FP)

    print("precision:", precision)

    # plt.bar(range(X_train.shape[1]), gb.feature_importances_)
    # plt.xticks(range(X_train.shape[1]), X.columns)

    # plt.figure(figsize=(6 * 1.618, 6))
    # index = np.arange(len(X.columns))
    # bar_width = 0.35
    # plt.barh(index, predictor.feature_importances_, color='black', alpha=0.5)
    # plt.ylabel('features')
    # plt.xlabel('importance')
    # plt.title('Feature importance: Accuracy:')
    # plt.yticks(index, X.columns)
    # plt.tight_layout()
    # plt.show()
    # print(predictor.coefs_)
    # print(predictor.get_params())
    return accuracy


if __name__ == "__main__":
    clf = MLPClassifier(solver='sgd', activation='tanh', alpha=1e-5, hidden_layer_sizes=(len(all_columns), 11, 11, 1),
                        random_state=1, max_iter=10000)
    predictor = clf

    dirname = os.path.dirname(__file__)
    dataset_source = os.path.join(dirname, "..\\team_selection\\final_dataset.csv")

    input_data = pd.read_csv(dataset_source)
    # input_data = input_data.sample(frac=1).reset_index(drop=True)

    X = input_data[all_columns].copy()
    # TODO do not scale toss, result, wickets match number, batting position, etc
    print(len(X.columns))
    y = input_data["result"]  # Labels
    X.to_csv("match_win.csv")
    # oversample = SMOTE()
    # X, y = oversample.fit_resample(X, y)

    scaler = preprocessing.StandardScaler().fit(X)
    data_scaled = scaler.transform(X)
    df = pd.DataFrame(data=data_scaled, columns=X.columns)
    pickle.dump(scaler, open(scaler_file, 'wb'))
    # df["runs_scored"] = X['runs_scored']
    # df["balls_faced"] = X['balls_faced']
    # df["batting_position"] = X['batting_position']

    X = df
    train_set = 2422
    train_set = 2077
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_train = X.iloc[:train_set, :]
    X_test = X.iloc[train_set + 1:, :]
    y_train = y.iloc[:train_set]
    y_test = y.iloc[train_set + 1:]

    select_features(X_train, y_train)

    predictor.fit(X_train, y_train)
    pickle.dump(predictor, open(model_file, 'wb'))
    win_predict(predictor)
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
