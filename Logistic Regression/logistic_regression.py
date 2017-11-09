from random import seed
from random import randrange
from csv import reader
from math import exp
import os


def load_csv(filename):
    dataset = list()
    with open(os.getcwd() + '/' + filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


def dataset_minmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax


def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# Sigmoid Function
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row) - 1):
        yhat += coefficients[i + 1] * row[i]
    return 1.0 / (1.0 + exp(-yhat))


# Estimate logistic regression coefficients using stochastic gradient descent (TODO Batch Gradient Descent)

'''
* Batch gradient descent computes the gradient using the whole dataset. This is great for convex, or relatively smooth error manifolds.
  In this case, we move somewhat directly towards an optimum solution, either local or global. Additionally, batch gradient descent, 
  given an annealed learning rate, will eventually find the minimum located in it's basin of attraction.

* Stochastic gradient descent (SGD) computes the gradient using a single sample. Most applications of SGD actually use a minibatch 
  of several samples, for reasons that will be explained a bit later. SGD works well (Not well, I suppose, but better than batch 
  gradient descent) for error manifolds that have lots of local maxima/minima. In this case, the somewhat noisier gradient calculated 
  using the reduced number of samples tends to jerk the model out of local minima into a region that hopefully is more optimal. Single 
  samples are really noisy, while minibatches tend to average a little of the noise out. Thus, the amount of jerk is reduced when using
  minibatches. A good balance is struck when the minibatch size is small enough to avoid some of the poor local minima, but large enough 
  that it doesn't avoid the global minima or better-performing local minima. (Incidently, this assumes that the best minima have a larger 
  and deeper basin of attraction, and are therefore easier to fall into.)
'''


def coefficients_sgd(train, l_rate, n_epoch):
    coef = [0.0 for i in range(len(train[0]))]
    for epoch in range(n_epoch):
        for row in train:
            yhat = predict(row, coef)
            error = row[-1] - yhat
            coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
            for i in range(len(row) - 1):
                coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
    return coef


def logistic_regression(train, test, l_rate, n_epoch):
    predictions = list()
    coef = coefficients_sgd(train, l_rate, n_epoch)
    for row in test:
        yhat = predict(row, coef)
        yhat = round(yhat)
        predictions.append(yhat)
    return (predictions)


seed(1)
# load and prepare data
filename = 'data.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)

minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)

n_folds = 5
l_rate = 0.1
n_epoch = 100

scores = evaluate_algorithm(dataset, logistic_regression, n_folds, l_rate, n_epoch)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
