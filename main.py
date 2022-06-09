# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import pandas as pd

def main():
    dataset = np.genfromtxt("data/BikeData.csv",delimiter=',')
    print(dataset.shape)
    print(max(dataset[1:,1]))
    print(min(dataset[1:,1]))
    outputs = np.sort(dataset[1:,1])
    print(outputs.shape)
    q1 = np.median(outputs)
    # Features: Date,Rented Bike Count,Hour,Temperature(°C),Humidity(%)
    # ,Wind speed (m/s),Visibility (10m),Dew point temperature(°C),Solar Radiation (MJ/m2)
    # ,Rainfall(mm),Snowfall (cm),Seasons,Holiday,Functioning Day
    ## so features: 0 can be dismissed, feature 1 is the output ##
    # have to decided how to split up classification: quartiles?

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
