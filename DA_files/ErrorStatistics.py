import numpy as np

def RMSE(model_data, observed_data):
    return np.sqrt(1/model_data.shape[1])*np.linalg.norm(model_data-observed_data, axis = 1)
    #return np.sqrt(1/model_data.shape[1]*np.sum((model_data-observed_data)**2, axis = 1))

def Bias(model_data, observed_data):
    return 1/model_data.shape[1]*np.sum(model_data-observed_data, axis = 1)

def InfNorm(model_data, observed_data):
    return np.max(np.abs(model_data-observed_data), axis = 1)

def OneNorm(model_data, observed_data):
    return 1/model_data.shape[1]*np.sum(np.abs(model_data-observed_data), axis = 1)