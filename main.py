# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import sklearn
from datetime import datetime

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
import statistics

def main():
    ## PREPROCESSING: UNCOMMENT IF RUNNING FOR FIRST TIME
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
    # x,y = dataset.shape
    # Y = np.asarray(Y, dtype=np.float64)
    # Y_sorted = np.sort(Y)
    # medians = []
    # k = 8
    # r = int(x / k)
    # for i in range(1, k+1):
    #     medians.append(np.median(Y_sorted[int(i - 1) * r:r * i]))
    # for i in range(x):
    #     value = medians[0]
    #     for med in medians:
    #         if value <= Y[i] <= med:
    #             value = med
    #     Y[i] = value
    # print(medians)
    # #SAVE PROCESSED DATA
    # np.save('x.txt', X)
    # np.save('y.txt', Y)
    #
    # # LOAD PROCESSED DATA
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
    x,y = Xtr.shape
    # step 1 Training, sliding window approach
    new_xtr = np.zeros((x,y+8))
    new_xtr[:,0:y] = Xtr
    for k in range(1,9):
        m = int(x/8)
        window_x = X[(k-1)*m:k*m,:]
        window_y = Y[(k-1)*m:k*m]
        rfc = RandomForestClassifier(max_depth=5, random_state=1, n_estimators=25).fit(Xtr,Ytr)
        etc = ExtraTreesClassifier(max_depth=5, random_state=1, n_estimators=25).fit(Xtr,Ytr)
        pred_a = rfc.predict_proba(window_x)
        pred_b = etc.predict_proba(window_x)
        x2,y2 = pred_a.shape
        new_feat = np.zeros((x2,y2))
        for i in range(x2):
            for j in range(y2):
                new_feat[i,j] = np.average([pred_a[i,j], pred_b[i,j]])
        new_xtr[(k-1)*m:k*m,y:] = new_feat

    cascade(Xtr,Ytr,new_xtr, Xte, Yte)

    return

    # plt.plot(range(100), pred1[0:100], label = 'Rand. Forest', linestyle = "-")
    # plt.plot(range(100), pred2[0:100], label = 'Extra Trees', linestyle = "--")
    # plt.title('Time Forecasting over Future hours')
    # plt.show()
    # plt.plot(range(x), pred_v2)
    # plt.title('Extra Trees Predicted Bike Rentals Over Date Time')
    # plt.show()

    correct = 0
    ddf_auc = correct/x1
    rf_auc = RandomForestClassifier(max_depth=25, random_state=1, n_estimators=100).fit(Xtr,Ytr).score(Xte,Yte)
    mlp_auc = MLPClassifier(random_state=1, max_iter=1000).fit(Xtr,Ytr).score(Xte,Yte)
    gbc_auc = GradientBoostingClassifier(random_state=1, max_depth=100).fit(Xtr,Ytr).score(Xte,Yte)
    auc = [ddf_auc, rf_auc, mlp_auc, gbc_auc]
    auc_models = ['DDF', 'RF', 'MLP', 'GBP']
    plt.bar(auc_models,auc, width= .5)
    plt.title('Training Accuracies of Popular Models Compared To DDF')
    plt.show()


def cascade(X, Y, Xtr, Xte, Yte):
    x, y = X.shape
    correct = 0;
    for i in range(x):
        # Step 2, Learn Class Distribution from random forests and extra tree classifiers
        train = X
        test = Xte[i]
        for j in range(8):
            rfc1 = RandomForestClassifier(max_depth=5, random_state=1, n_estimators=25).fit(train,Y)
            pred1 = rfc1.predict_proba([test])
            rfc2 = RandomForestClassifier(max_depth=5, random_state=1, n_estimators=25).fit(train,Y)
            pred2 = rfc2.predict_proba([test])
            etc1 = ExtraTreesClassifier(max_depth=5, random_state=1, n_estimators=25).fit(train,Y)
            pred3 = etc1.predict_proba([test])
            etc2 = ExtraTreesClassifier(max_depth=5, random_state=1, n_estimators=25).fit(train,Y)
            pred4 = etc2.predict_proba([test])
            feat = np.zeros(8)
            for k in range(8):
                feat[k] = np.mean([pred1[0,k],pred2[0,k],pred3[0,k],pred4[0,k]])
            test = np.zeros(18)
            test[0:10] = Xte[i]
            test[10:] = feat
            train = Xtr
            if j == 7:
                prediction = statistics.mode([rfc1.predict([test])[0],rfc2.predict([test])[0],etc1.predict([test])[0],etc2.predict([test])[0]])
                if prediction == Yte[i]:
                    correct+=1
    print(correct/len(Yte))
    drf_auc = correct/len(Yte)
    rf_auc = RandomForestClassifier(max_depth=5, random_state=1, n_estimators=25).fit(Xtr,Ytr).score(Xte,Yte)
    mlp_auc = MLPClassifier(random_state=1, max_iter=500).fit(Xtr, Ytr).score(Xte, Yte)
    gbc_auc = GradientBoostingClassifier(max_depth=5, random_state=1, n_estimators=25).fit(Xtr, Ytr).score(Xte, Yte)
    aucs = [drf_auc,rf_auc,mlp_auc,gbc_auc]
    auc_models = ['DRF','RF','MLP','GBC']
    plt.bar(auc_models, aucs, width=.5)
    plt.tile('Accuracy of models for dataet')
    plt.show()




def minmaxmediant(A):
    print('Min:', np.min(A))
    print('Median:', np.median(A))
    print('Max:', np.max(A))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
