from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


def get_error_curves(X_train, y_train, X_test, y_test, output_columns, max_iters, test_attr):
    RMSE_train_array = []
    RMSE_test_array = []
    R2_test_array = []
    R2_train_array = []
    iter_range = range(1, max_iters)
    for n_i in iter_range:
        print(n_i)
        depth = n_i
        estimators = 200
        predictor = RandomForestRegressor(max_depth=depth, n_estimators=estimators, random_state=1, max_features="auto",
                                          n_jobs=-1)
        predictor.fit(X_train, y_train)
        train_pred = pd.DataFrame(predictor.predict(X_train), columns=output_columns)
        test_pred = pd.DataFrame(predictor.predict(X_test), columns=output_columns)

        rmse_train = np.sqrt(metrics.mean_squared_error(y_train[test_attr], train_pred[test_attr]))
        rmse_test = np.sqrt(metrics.mean_squared_error(y_test[test_attr], test_pred[test_attr]))

        r2_train = metrics.r2_score(y_train[test_attr], train_pred[test_attr])
        r2_test = metrics.r2_score(y_test[test_attr], test_pred[test_attr])

        RMSE_train_array.append(rmse_train)
        RMSE_test_array.append(rmse_test)

        R2_train_array.append(r2_train)
        R2_test_array.append(r2_test)

    # plt.plot(iter_range, RMSE_test_array, color='blue', label="test_data")
    # plt.plot(iter_range, RMSE_train_array, color='red', label="train_data")
    # plt.title('RMSE vs Depth')
    # plt.xlabel('Depth')
    # plt.ylabel('RMSE')
    # plt.legend()
    # plt.show()

    plt.plot(iter_range, R2_test_array, color='blue', label="test_data")
    plt.plot(iter_range, R2_train_array, color='red', label="train_data")
    plt.title('R2 vs Depth')
    plt.xlabel('Depth')
    plt.ylabel('R2')
    plt.legend()
    plt.show()
