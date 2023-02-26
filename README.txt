A class project building a perceptron and gradient descent from scratch.

perceptron.py

perceptron_train(X, Y)
The function perceptron_train takes X, a collection samples, with any number of real valued feature vectors in any dimension and Y the corresponding set of labels for those feature vectors.

The function runs through a set number of epochs (100 by default). In each epoch it runs thru each sample calculating the activation for that sample.  If the activation multiplied by its label is less than or equal to 0, then the weights and the bias are updated and it keeps running activations with the updated values.  Upon completion the final w and b are returned


perceptron_test(X_test, Y_test, w, b)
This function takes X_test, a collection samples, with any number of real valued feature vectors in any dimension, Y_test, the corresponding set of labels for those feature vectors, w, an array of weights from a trained perceptron model corresponding to the number of features from X_test, and b, the bias value from a trained perceptron model.

The function runs through each sample in X_test calculates the activation for it.  If the sign of the activation matches the label then it has predicted correctly for that sample.  It counts the number of correct predictions and divides by the total to get accuracy.  Finally it returns the accuracy.



gradient_descent(gradient_of_f, init_x, learning_rate)

Parameters

- gradient_of_f: The gradient function of the function you want to minimize.

- init_x: A numpy array containing the x values for where you want to start your descent. As many x's as your problem has dimensions. I.e. [x_1, x_2, ... , x_n]

- learning_rate: The size of the step taken each iteration.

Hyperparameters

- completion_value: (default: 0.0001) Once the magnitude of the x's goes below this value, the gradient descent is terminated

Description

The function gradient_descent goes thru a series of iterations taking a step the size of the learning rate towards the minimum of the passed function.

First the gradient is calculated for the current location.

Then the location is update by the learning_rate (step size).

Then the magnitude of the x vector is calculate.

If it hasn't reach the completion value it continues.
