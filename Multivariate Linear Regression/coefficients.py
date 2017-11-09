def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row) - 1):
        yhat += coefficients[i + 1] * row[i]
    return yhat


# Use SGD to reach the global optima for the Cost Function
def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            yhat = predict(row, coef)
            error = yhat - row[-1]
            sum_error += error ** 2
            coef[0] = coef[0] - l_rate * error
            for i in range(len(row) - 1):
                coef[i + 1] = coef[i + 1] - l_rate * error * row[i]
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    return coef


dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
# Learning rate
l_rate = 0.001

# Iterations
n_epoch = 500000
coef = coefficients_sgd(dataset, l_rate, n_epoch)
print(coef)
