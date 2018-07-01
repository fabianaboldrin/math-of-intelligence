# good exemple of calcutaling m and b with gradient descent
# in this exemple, x = distance a person bikes and y = amout of calories they lost.
# also, m and b are values of the following equation: y = m.x + b

import numpy as np

# m is slope, b is y-intercept

def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i n range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))