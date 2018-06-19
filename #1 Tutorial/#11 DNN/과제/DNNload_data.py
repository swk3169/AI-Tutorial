import pandas as pd
import numpy as np

def load_data():

    data = pd.read_excel("C:\Temp\\training.xlsx", sep=',');

    attributes = ["class", "GLCM_pan", "Mean_Green", "Mean_Red", "Mean_NIR", "SD_pan"]
    input_attributes = ["class", "GLCM_pan", "Mean_Green", "Mean_Red", "Mean_NIR", "SD_pan"]
    data.columns = attributes
    data["class"] = data["class"].map({"w": 1, "n": 0})
    data = data.reindex(np.random.permutation(data.index))

    print(data)

    train_size = int(len(data) * 0.8)
    train_set = data[:train_size];
    print(len(train_set))
    test_set = data[train_size:];
    print(len(test_set))

    x_train, x_test, y_train, y_test = train_set[input_attributes], test_set[input_attributes], train_set['class'], \
                                       test_set['class']

    return (x_train, y_train), (x_test, y_test)