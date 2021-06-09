import pandas as pd
import os
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from final_data.encoders import *
import pickle
from sklearn.metrics import accuracy_score

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
    # 'season',
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
    # 'match_number',
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
    "economy"
]


def predict_for_team(input_team_data):
    team_data = input_team_data[all_columns].copy()
    loaded_predictor = pickle.load(open(model_file, 'rb'))
    loaded_scaler = pickle.load(open(scaler_file, 'rb'))
    # team_performance = loaded_scaler.transform(team_data)
    predicted = loaded_predictor.predict_proba(team_data)
    df = pd.DataFrame(predicted, columns=["lose", "win"])
    team_data["winning_probability"] = df["win"].to_numpy()
    return team_data, df["win"].mean()


def win_predict(predictor):
    y_pred = predictor.predict(X_test)
    # probability = predictor.predict_proba(X_test)
    # print(type(y_pred), type(y_test))
    # print(type(probability))
    # for i, row in y_test.items():
    #     index=i-(train_set+1)
    #     print(i, row, y_pred[index], probability[index])

    # for classifiers
    # print("Score:", predictor.score(X_test, y_test))
    accuracy = metrics.accuracy_score(y_test, y_pred)
    # cvs = cross_val_score(predictor, X, y, scoring='accuracy', cv=10).mean()
    print("Accuracy:", accuracy)
    # print("Cross Validation Score:", cvs)
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

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
    clf = MLPClassifier(solver='sgd', activation='tanh', alpha=1e-5, hidden_layer_sizes=(len(all_columns), 11, 1),
                        random_state=1, max_iter=10000)
    gb = GradientBoostingClassifier(n_estimators=1000)
    predictor = clf

    dirname = os.path.dirname(__file__)
    dataset_source = os.path.join(dirname, "..\\team_selection\\final_dataset.csv")

    input_data = pd.read_csv(dataset_source)
    # input_data = input_data.sample(frac=1).reset_index(drop=True)

    X = input_data[all_columns].copy()
    X['runs_scored'] = X['runs_scored'].apply(lambda x: encode_runs(x))
    X['balls_faced'] = X['balls_faced'].apply(lambda x: encode_balls_faced(x))
    X['fours_scored'] = X['fours_scored'].apply(lambda x: encode_fours(x))
    X['sixes_scored'] = X['sixes_scored'].apply(lambda x: encode_sixes(x))
    X['batting_position'] = X['batting_position'].apply(lambda x: encode_batting_position(x))
    X['economy'] = X['economy'].apply(lambda x: encode_econ(x))
    X['deliveries'] = X['deliveries'].apply(lambda x: encode_deliveries_bowled(x))
    X['wickets_taken'] = X['wickets_taken'].apply(lambda x: encode_wickets(x))
    # TODO do not scale toss, result, wickets match number, batting position, etc
    print(len(X.columns))
    y = input_data["result"]  # Labels
    X.to_csv("match_win.csv")
    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)

    scaler = preprocessing.StandardScaler().fit(X)
    data_scaled = scaler.transform(X)
    df = pd.DataFrame(data=data_scaled, columns=X.columns)
    pickle.dump(scaler, open(scaler_file, 'wb'))
    # df["runs_scored"] = X['runs_scored']
    # df["balls_faced"] = X['balls_faced']
    # df["batting_position"] = X['batting_position']

    X = df
    print(X)

    train_set = 2300
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    X_train = X.iloc[:train_set, :]
    X_test = X.iloc[train_set + 1:, :]
    y_train = y.iloc[:train_set]
    y_test = y.iloc[train_set + 1:]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    #

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
