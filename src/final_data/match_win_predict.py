import pandas as pd
import os
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

clf = MLPClassifier(solver='sgd', activation='tanh', alpha=1e-5, hidden_layer_sizes=(43, 11, 1), random_state=1,
                    max_iter=10000)
predictor = clf

dirname = os.path.dirname(__file__)
dataset_source = os.path.join(dirname, "..\\team_selection\\final_dataset.csv")

input_data = pd.read_csv(dataset_source)
X = input_data[[
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
    "econ"
]]
print(len(X.columns))
y = input_data["result"]  # Labels
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)

scaler = preprocessing.StandardScaler().fit(X)
data_scaled = scaler.transform(X)
X = pd.DataFrame(data=data_scaled, columns=X.columns)
train_set = 2423

X_train = X.iloc[:train_set, :]
X_test = X.iloc[train_set + 1:, :]
y_train = y.iloc[:train_set]
y_test = y.iloc[train_set + 1:]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#
predictor.fit(X_train, y_train)


def batting_predict(predictor):
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
    print("Accuracy:", accuracy)
    # print("Cross Validation Score:", cross_val_score(predictor, X, y, scoring='accuracy', cv=10).mean())
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    # plt.figure(figsize=(6 * 1.618, 6))
    # index = np.arange(len(X.columns))
    # bar_width = 0.35
    # plt.barh(index, predictor.feature_importances_, color='black', alpha=0.5)
    # plt.ylabel('features')
    # plt.xlabel('importance')
    # plt.title('Feature importance')
    # plt.yticks(index, X.columns)
    # plt.tight_layout()
    # plt.show()
    # print(predictor.coefs_)
    # print(predictor.get_params())
    return accuracy


def predict_for_team(team_data):
    print(set(X.columns) - set(team_data.columns))
    team_performance = team_data.copy()
    predicted = predictor.predict_proba(team_performance)
    df = pd.DataFrame(predicted, columns=["lose", "win"])
    team_performance["winning_probability"] = df["win"].to_numpy()
    return team_performance, df["win"].mean()


if __name__ == "__main__":
    batting_predict(predictor)
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
