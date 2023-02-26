import numpy as np
import math

def gradient_descent(gradient_of_f, init_x, learning_rate):
    completion_value = 0.0001
    dimensions = init_x.size
    x = np.array(init_x)
    i = 0
    mag_grad = 1
    while mag_grad > completion_value:
        gradient_at_x = gradient_of_f(x)
        x = x - learning_rate * gradient_at_x   
        mag_grad = np.linalg.norm(gradient_at_x)
        i += 1
    return x