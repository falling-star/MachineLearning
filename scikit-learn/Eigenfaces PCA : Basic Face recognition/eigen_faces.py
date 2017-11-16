"""

X Shape :  (1140, 2914)
Total dataset size:
n_samples: 1140 62 47
n_features: 2914
n_classes: 5

Best estimator found by grid search:
SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.005, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

                   precision    recall  f1-score   support

     Colin Powell       0.94      0.91      0.92        64
  Donald Rumsfeld       0.92      0.72      0.81        32
    George W Bush       0.87      0.97      0.91       127
Gerhard Schroeder       0.93      0.86      0.89        29
       Tony Blair       0.90      0.79      0.84        33

      avg / total       0.90      0.89      0.89       285


[[ 58   0   4   0   2]
 [  2  23   6   1   0]
 [  2   1 123   0   1]
 [  0   0   4  25   0]
 [  0   1   5   1  26]]


"""

from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

lfw_people = fetch_lfw_people(min_faces_per_person=100)

n_samples, h, w = lfw_people.images.shape

X = lfw_people.data

print("X Shape : ", X.shape)

# How many pixels in each image
n_features = X.shape[1]

y = lfw_people.target
target_names = lfw_people.target_names
# number of different people
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples, h, w)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

# Number of Top Eigenfaces after PCA is applied
n_components = 150

pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)

# Eigenfaces : top n_components
eigenfaces = pca.components_.reshape((n_components, h, w))

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

# finds best C & gamma parameter estimates for SVC classifier
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("Best estimator found by grid search:")
print(clf.best_estimator_)

# predict test set
y_pred = clf.predict(X_test_pca)

# calculate precision and recall with f1 scope
print(classification_report(y_test, y_pred, target_names=target_names))
# calculate confusion matrix
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))


"""

X Shape :  (1140, 2914)
Total dataset size:
n_samples: 1140 62 47
n_features: 2914
n_classes: 5

Best estimator found by grid search:
SVC(C=1000.0, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.005, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
  
                   precision    recall  f1-score   support

     Colin Powell       0.94      0.91      0.92        64
  Donald Rumsfeld       0.92      0.72      0.81        32
    George W Bush       0.87      0.97      0.91       127
Gerhard Schroeder       0.93      0.86      0.89        29
       Tony Blair       0.90      0.79      0.84        33

      avg / total       0.90      0.89      0.89       285


[[ 58   0   4   0   2]
 [  2  23   6   1   0]
 [  2   1 123   0   1]
 [  0   0   4  25   0]
 [  0   1   5   1  26]]


"""
