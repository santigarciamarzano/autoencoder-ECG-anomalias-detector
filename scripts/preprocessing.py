import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_data(train_path, test_path):
    
    train_data = pd.read_csv(train_path, header=None)
    test_data = pd.read_csv(test_path, header=None)
    
    return train_data, test_data


"""def preprocess_data(train_data, test_data):
   
    x_train = train_data.iloc[:, 1:].values
    x_test = test_data.iloc[:, 1:].values
    
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    
    return x_train_scaled, x_test_scaled, scaler"""

def preprocess_data(train_data, test_data):
    scaler = MinMaxScaler()

    if train_data is not None:
        x_train = train_data.iloc[:, 1:].values
        x_train_scaled = scaler.fit_transform(x_train)
    else:
        x_train_scaled = None

    x_test = test_data.iloc[:, 1:].values

    if x_train_scaled is not None:
        x_test_scaled = scaler.transform(x_test)
    else:
        x_test_scaled = scaler.fit_transform(x_test)

    return x_train_scaled, x_test_scaled, scaler