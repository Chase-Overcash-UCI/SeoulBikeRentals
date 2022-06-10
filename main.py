# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import sklearn
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import statistics

def main():
    # ## PREPROCESSING: UNCOMMENT IF RUNNING FOR FIRST TIME
    # dataset = np.genfromtxt("data/BikeData.csv", delimiter=',', dtype='str')[1:, :]
    # x, y = dataset.shape
    # X = np.zeros((x, y - 4))
    # Y = dataset[:, 1]
    # for j in [2, 3, 4, 5, 6, 9, 11, 12]:
    #     if j == 2:
    #         # HOUR FEATURE
    #         for i in range(x):
    #             hour = int(dataset[i, j])
    #             # if 0 <= hour < 6:
    #             #     X[i, 0] = 0
    #             # elif 6 <= hour < 12:
    #             #     X[i, 0] = 1
    #             # elif 12 <= hour < 18:
    #             #     X[i, 0] = 2
    #             # else:
    #             #     X[i, 0] = 3
    #             X[i,0] = hour
    #     elif j == 3:
    #         # TEMPERATURE FEATURE
    #         for i in range(x):
    #             temp = float(dataset[i, j])
    #             if temp < 14:
    #                 X[i, 1] = 0
    #             else:
    #                 X[i, 1] = 1
    #     elif j == 4:
    #         # HUMIDITY FEATURE
    #         for i in range(x):
    #             humidity = float(dataset[i, j])
    #             if humidity < 25:
    #                 X[i,2] = 0
    #             elif 25 <= humidity < 50:
    #                 X[i, 2] = 1
    #             elif 50 <= humidity < 75:
    #                 X[i, 2] = 2
    #             else:
    #                 X[i,2] = 3
    #     elif j == 5:
    #         # WIND SPEED FEATURE
    #         for i in range(y):
    #             wind = float(dataset[i, j])
    #             if wind <= 1.5:
    #                 X[i, 3] = 0
    #             else:
    #                 X[i, 3] = 1
    #     elif j == 6:
    #         #print(minmaxmediant(np.asarray(dataset[:, j], dtype=np.float64)))
    #         # VISIBILITY FEATURE
    #         for i in range(x):
    #             visibility = float(dataset[i, j])
    #             if visibility <= 1000:
    #                 X[i, 4] = 0
    #             else:
    #                 X[i, 4] = 1
    #     elif j == 9:
    #         # PERCIPITATION FEATURE
    #         rainfall = np.asarray(dataset[:,j],dtype=np.float64)
    #         snowfall = np.asarray(dataset[:,j+1],dtype=np.float64)
    #         for i in range(x):
    #             percip = rainfall[i] + snowfall[i]
    #             if percip <= 0:
    #                 X[i, 5] = 0
    #             else:
    #                 X[i, 5] = 1
    #     elif j == 11:
    #         # SEASON FEATURE
    #         for i in range(x):
    #             season = dataset[i, j]
    #             if season == 'Winter':
    #                 X[i, 6] = 0
    #             elif season == 'Spring':
    #                 X[i, 6] = 1
    #             elif season == 'Summer':
    #                 X[i, 6] = 2
    #             elif season == 'Autumn':
    #                 X[i, 6] = 3
    #     elif j == 12:
    #         # HOLIDAY/ WEEKEND / FUNCTION DAY FEATURE
    #         for i in range(x):
    #             holiday = dataset[i, j]
    #             functioning = dataset[i, j + 1]
    #             if holiday == 'Holiday' or functioning == 'No':
    #                 X[i, 7] = 1
    #             else:
    #                 X[i, 7] = 0
    #             day, month, year = str.split(dataset[i, 0], '/')
    #             date = pd.Timestamp(year=int(year), month=int(month), day=int(day))
    #             weekday = date.day_of_week
    #             X[i,8] = weekday
    #
    # Y = np.asarray(Y, dtype=np.float64)
    # Y_sorted = np.sort(Y)
    # medians = []
    # k = 8
    # r = int(x / k)
    # for i in range(1, k+1):
    #     medians.append(np.median(Y_sorted[int(i - 1) * r:r * i]))
    # for i in range(x):
    #     for med in medians:
    #         if Y[i] <= med:
    #             Y[i] = med
    #
    # # SAVE PROCESSED DATA
    # np.save('x.txt', X)
    # np.save('y.txt', Y)

    X = np.load('x.txt.npy')
    Y = np.load('y.txt.npy')
    x,y = X.shape
    ## Split Data into Training and Testing
    UB = int(x * .9)
    Xtr = X[:UB, :]
    Ytr = Y[:UB]
    # for i in range(UB):
    #     print(Xtr[i,:])
    #     print(Ytr[i])
    # print(Ytr)
    Xte = X[UB:, :]
    Yte = Y[UB:]

    # step 1 call cascade on Xtr, Xte
    updated_Xtr,_ = cascade(Xtr, Ytr, Xte, False)

    # Step 2 Sliding Window Approach
    windows_feat = []
    windows_class = []
    for i in range(1,4):
        m = int(x/4)
        window_x = X[(i-1)*m:i*m,:]
        window_y = Y[(i-1)*m:i*m]
        rfc = RandomForestClassifier(max_depth=100, random_state=1, n_estimators=500).fit(Xtr,Ytr)
        etc = ExtraTreesClassifier(max_depth=100, random_state=1, n_estimators=500).fit(Xtr, Ytr)
        pred_v1 = rfc.predict(window_x)
        probas_v1 = rfc.predict_proba(window_x)
        windows_feat.append(probas_v1 * window_x)
        pred_v2 = etc.predict(Xte)
        probas_v2 = etc.predict_proba(window_x)
        windows_feat.append(probas_v2 * window_x)
        windows_class.append(pred_v1)
        windows_class.append(pred_v2)

    # Step 3 Training Layer by Layer.
    for z in range(3):
        # Call Cascade at each layer
        for k in range(4):
            updated_Xtr = cascade(updated_Xtr, Ytr, window_feat[k], False)

    final_rfc = RandomForestClassifier.fit(updated_Xtr,Ytr)
    final_rfc.score(Xte)


def cascade(Xtr, Ytr, Xte, predict):
    # Step 2, Learn Class Distribution from random forests and extra tree classifiers
    rfc1 = RandomForestClassifier(max_depth=20, random_state=1, n_estimators=100).fit(Xtr,Ytr)
    feats1 = rfc1.predict_proba(Xte)
    rfc2 = RandomForestClassifier(max_depth=20, random_state=1, n_estimators=100).fit(Xtr,Ytr)
    feats2 = rfc2.predict_proba(Xte)
    etc1 = ExtraTreesClassifier(max_depth=20, random_state=1, n_estimators=100).fit(Xtr,Ytr)
    feats3 = etc1.predict_proba(Xte)
    etc2 = ExtraTreesClassifier(max_depth=20, random_state=1, n_estimators=100).fit(Xtr,Ytr)
    feats4 = etc2.predict_proba(Xte)
    x,y = Xtr.shape
    updated = np.zeros((x,y))
    x1,y1 = Xte.shape
    for i in range(y):
        class_val = (feats1[i] + feats2[i] + feats3[i] + feats4[i]) / 4
        updated[:, i] = class_val * X[:,i]
    if predict:
        pred1 = rfc1.predict(Xte)
        pred2 = rfc2.predict(Xte)
        pred3 = etc1.predict(Xte)
        pred4 = etc2.predict(Xte)

        new_predictions = np.zeros(x1)
        for i in range(x1):
            new_predictions[i] = statistics.mode([pred1[i], pred2[i], pred3[i],pred4[i]])
        return updated, new_predictions
    return updated, None

def minmaxmediant(A):
    print('Min:', np.min(A))
    print('Median:', np.median(A))
    print('Max:', np.max(A))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
