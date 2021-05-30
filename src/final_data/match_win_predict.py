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
from sklearn import preprocessing

RF = RandomForestClassifier(n_estimators=100, criterion='entropy', bootstrap=False, max_depth=100,
                            class_weight={0: 1, 1: 1})
gnb = GaussianNB()
clf = MLPClassifier(solver='lbfgs', activation='relu', alpha=1e-5, hidden_layer_sizes=(5, 6), random_state=1,
                    max_iter=10000)
SVM = svm.SVC(kernel='linear', C=1)

dirname = os.path.dirname(__file__)
dataset_source = os.path.join(dirname, "output\overall.csv")


def batting_predict(predictor):
    input_data = pd.read_csv(dataset_source)

    scaler = preprocessing.StandardScaler().fit(input_data)
    data_scaled = scaler.transform(input_data)
    final_df = pd.DataFrame(data=data_scaled, columns=input_data.columns)
    # input_data.reset_index(drop=True, inplace=True)
    final_df["result"] = input_data["result"]
    input_data = final_df

    # X = input_data.loc[:, input_data.columns != 'result']
    X = input_data[[
        # 'player_id',
        # 'runs_scored',
        # 'balls_faced',
        # 'fours_scored',
        # 'sixes_scored',
        # 'strike_rate',
        # 'batting_position',
        # 'overs_bowled',
        # 'deliveries',
        # 'maidens',
        # 'runs_conceded',
        # 'wickets_taken',
        # 'dots',
        # 'fours_given',
        # 'sixes_given',
        # 'econ',
        # 'wides',
        # 'no_balls',
        'score',
        'wickets',
        # 'overs',
        'balls',
        # 'inning',
        # 'opposition_id',
        # 'venue_id',
        # 'toss',
        # 'season_id',
        # 'match_number',
        # 'batting_temp',
        # 'batting_feels',
        # 'batting_wind',
        # 'batting_gust',
        # 'batting_rain',
        # 'batting_humidity',
        # 'batting_cloud',
        # 'batting_pressure',
        # 'bowling_temp',
        # 'bowling_feels',
        # 'bowling_wind',
        # 'bowling_gust',
        # 'bowling_rain',
        # 'bowling_humidity',
        # 'bowling_cloud',
        # 'bowling_pressure',
        # 'batting_contribution',
        # 'bowling_contribution'
    ]]  # Features
    y = input_data["result"]  # Labels
    print(X.columns)
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
    print("Cross Validation Score:", cross_val_score(predictor, X, y, cv=10).mean())
    print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

    # print(predictor.coefs_)
    # print(predictor.get_params())
    return accuracy


batting_predict(RF)
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
