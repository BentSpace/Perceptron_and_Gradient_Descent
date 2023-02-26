import numpy as np
import math

def perceptron_train(X, Y):
    num_rows, num_cols = X.shape
    num_features = num_cols
    num_samples = num_rows
    w = np.zeros(num_features)
    b = 0
    max_epochs = 100
    epoch = 1
    while epoch < max_epochs:
        sample_i = 0
        while sample_i < num_samples:
            feature_i = 0
            activation = b
            while feature_i < num_features:
                activation += w[feature_i] * X[sample_i][feature_i]
                feature_i += 1
            update = Y[sample_i] * activation
            if update <= 0:
                feature_i = 0
                while feature_i < num_features:
                    w[feature_i] = w[feature_i] + Y[sample_i] * X[sample_i][feature_i]  
                    feature_i += 1
                b = b + Y[sample_i]
            sample_i += 1           
        epoch += 1
    return w, b

def perceptron_test(X_test, Y_test, w, b):
    num_rows, num_cols = X_test.shape
    num_features = num_cols
    num_samples = num_rows
    sample_i = 0
    correct_predictions = 0
    incorrect_predictions = 0
    while sample_i < num_samples:
        feature_i = 0
        activation = b
        while feature_i < num_features:
            activation += w[feature_i] * X_test[sample_i][feature_i]
            feature_i += 1
        if activation < 0:
            prediction = -1
        else:
            prediction = 1
            
        if prediction == Y_test[sample_i]:
            correct_predictions += 1
        else:
            incorrect_predictions += 1

        sample_i += 1  
    accuracy = correct_predictions / (correct_predictions + incorrect_predictions)
    return accuracy
