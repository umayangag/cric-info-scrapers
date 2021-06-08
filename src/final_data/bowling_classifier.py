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
import pickle

model_file = "bowling_performance_predictor.sav"

input_columns = [
    "bowling_consistency",
    "bowling_form",
    "bowling_temp",
    "bowling_wind",
    # "bowling_rain",
    "bowling_humidity",
    "bowling_cloud",
    "bowling_pressure",
    # "bowling_viscosity",
    # "batting_inning",
    # "bowling_session",
    # "toss",
    "bowling_venue",
    "bowling_opposition",
    "season",
]


def predict_bowling(dataset):
    loaded_predictor = pickle.load(open(model_file, 'rb'))
    predicted = loaded_predictor.predict(dataset[input_columns])
    result = pd.DataFrame(predicted, columns=y.columns)
    for column in y.columns:
        dataset[column] = result[column]
    dataset["runs_conceded"] = dataset.apply(lambda row: calculate_runs_conceded(row), axis=1)
    return dataset


def calculate_runs_conceded(row):
    if row["bowling_consistency"] == 0:
        return 0
    return decode_value(hash_economy, row["economy"]) * decode_value(hash_deliveries, row["deliveries"]) / 6


def bowling_predict_test():
    y_pred = predictor.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=output_bowling_columns)
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

    for attribute in output_bowling_columns:
        print("Accuracy:" + attribute, metrics.accuracy_score(y_test[attribute], y_pred[attribute]))
        print(attribute + '\n', confusion_matrix(y_test[attribute], y_pred[attribute], labels=[0, 1, 2, 3, 4]))

    # print("Cross Validation Score:", cross_val_score(predictor, X, y, cv=10).max())


if __name__ == "__main__":
    RF = RandomForestClassifier(n_estimators=1000, criterion='entropy', max_depth=1000,
                                # class_weight=[{0: 1, 1: 1, 2: 1}, {0: 1, 1: 1}, {0: 1, 1: 1, 2: 1}]
                                class_weight="balanced"
                                )
    predictor = RF

    dirname = os.path.dirname(__file__)
    dataset_source = os.path.join(dirname, "output\\bowling_encoded.csv")

    input_data = pd.read_csv(dataset_source)
    input_data = input_data.sample(frac=1).reset_index(drop=True)
    training_input_columns = input_bowling_columns.copy()
    training_input_columns.remove("player_name")

    X = input_data[input_columns]
    y = input_data[output_bowling_columns].copy()  # Labels
    y["economy"] = y["economy"].apply(encode_econ)
    y["deliveries"] = y["deliveries"].apply(encode_deliveries_bowled)
    y["wickets_taken"] = y["wickets_taken"].apply(encode_wickets)

    tempX = X.copy()
    tempX["deliveries"] = y["deliveries"]
    tempX["wickets_taken"] = y["wickets_taken"]

    oversample = SMOTE()
    tempX, runs_predicted = oversample.fit_resample(tempX, y["economy"])

    tempX["economy"] = runs_predicted
    X = tempX[input_columns]
    y = tempX[output_bowling_columns]

    # scaler = preprocessing.StandardScaler().fit(X)
    # data_scaled = scaler.transform(X)
    # final_df = pd.DataFrame(data=data_scaled, columns=X.columns)
    # X = final_df

    train_set = 1465
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # X_train = X.iloc[:train_set, :]
    # X_test = X.iloc[train_set + 1:, :]
    # y_train = y.iloc[:train_set]
    # y_test = y.iloc[train_set + 1:]
    predictor.fit(X_train, y_train)
    pickle.dump(predictor, open(model_file, 'wb'))
    bowling_predict_test()
