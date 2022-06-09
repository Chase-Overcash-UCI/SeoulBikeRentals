# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd
import sklearn as skl
from datetime import datetime

def main():
    ## PREPROCESSING
    dataset = np.genfromtxt("data/BikeData.csv",delimiter=',', dtype=str)[1:,:]
    x,y = dataset.shape
    X = np.zeros((x,y-5))
    Y = dataset[:,1]
    for j in [2,3,4,5,6,9,12]:
        if j == 2:
            # HOUR FEATURE
            for i in range(y):
                hour = int(dataset[i,j])
                if 0 <= hour < 6:
                    X[i, 0] = 0
                elif 6 <= hour < 12:
                    X[i,0] = 1
                elif 12 <= hour < 18:
                    X[i,0] = 2
                else: X[i,0] = 3
        elif j == 3:
            # TEMPERATURE FEATURE
            for i in range(y):
                temp = float(dataset[i,j])
                if temp < 14:
                    X[i,1] = 0
                else: X[i,1] = 1
        elif j == 4:
            # HUMIDITY FEATURE
            for i in range(y):
                humidity = float(dataset[i,j])
                if humidity < 50:
                    X[i,2] = 0
                else: X[i,2] = 1
        elif j == 5:
            # WIND SPEED FEATURE
            for i in range(y):
                wind = float(dataset[i,j])
                if wind <= 1.5:
                    X[i,3] = 0
                else: X[i,3] = 1
        elif j == 6:
            # VISIBILITY FEATURE
            for i in range(y):
                visibility = float(dataset[i,j])
                if visibility <= 1000:
                    X[i,4] = 0
                else: X[i,4] = 1
        elif j == 9:
            # PERCIPITATION FEATURE
            percip = float(dataset[i,j]) + float(dataset[i,j+1])
            if percip <=0:
                X[i,5] = 0
            else: X[i,5] = 1
        elif j == 11:
            # SEASON FEATURE
            for i in range(y):
                season = dataset[i,j]
                if season == 'Winter':
                    X[i,6] = 0
                elif season == 'Spring':
                    X[i,6] = 1
                elif season == 'Summer':
                    X[i,6] = 2
                elif season == 'Autumn':
                    X[i,6] = 3
        elif j == 12:
            # HOLIDAY/ WEEKEND / FUNCTION DAY FEATURE
            for i in range(y):
                day,month, year = str.split(dataset[i, 0], '/')
                date = pd.Timestamp(year = int(year),month = int(month), day = int(day))
                weekday = date.day_of_week
                holiday = dataset[i,j]
                functioning = dataset[i,j+1]
                if 4 < weekday <= 6 or holiday == 'Holiday' or functioning == 'No':
                    X[i,7] = 1
                else: X[i,7] = 0

    Y = np.asarray(Y, dtype= np.float64)
    Y_sorted = np.sort(Y)
    medians = []
    for i in range(1,9):
        medians.append(np.median(Y_sorted[(i-1)*1095:1095*i]))
    for i in range(y):
        min = 1000000
        mindex = -1
        for med in medians:
            diff = np.abs(med - Y[i])
            if diff < min:
                min = diff
                mindex = med
        Y[i] = mindex

    # SAVE PROCESSED DATA
    np.save('x.txt', X)
    np.save('y.txt', Y)

    ## Split Data into Training and Testing
    UB = int(x*.9)
    Xtr = X[:UB,:]
    Ytr = Y[:UB]
    Xte = X[UB:,:]
    Yte = Y[UB:]

def minmaxmediant(A):
    print('Min:', np.min(A))
    print('Median:', np.median(A))
    print('Max:', np.max(A))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
