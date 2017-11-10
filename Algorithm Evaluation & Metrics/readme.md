* k-fold Cross-Validation Split

A limitation of using the train and test split method is that you get a noisy estimate of algorithm performance.
The k-fold cross-validation method (also called just cross-validation) is a resampling method that provides a more
accurate estimate of algorithm performance. It does this by first splitting the data into k groups.
The algorithm is then trained and evaluated k times and the performance summarized by taking the mean performance score.
 Each group of data is called a fold, hence the name k-fold cross-validation. It works by first training the algorithm
 on the k-1 groups of the data and evaluating it on the kth hold-out group as the test set. This is repeated so that
 each of the k groups is given an opportunity to be held out and used as the test set. As such, the value of k should
 be divisible by the number of rows in your training dataset, to ensure each of the k groups has the same number of rows.




* How to Choose a Resampling Method

The gold standard for estimating the performance of machine learning algorithms on new data is k-fold cross-validation.
When well-configured, k-fold cross-validation gives a robust estimate of performance compared to other methods such as
the train and test split. The downside of cross-validation is that it can be time-consuming to run, requiring k different
models to be trained and evaluated. This is a problem if you have a very large dataset or if you are evaluating a model
that takes a long time to train.

The train and test split resampling method is the most widely used. This is because it is easy to understand and implement,
and because it gives a quick estimate of algorithm performance. Only a single model is constructed and evaluated. Although
the train and test split method can give a noisy or unreliable estimate of the performance of a model on new data, this
becomes less of a problem if you have a very large dataset.
Large datasets are those in the hundreds of thousands or millions of records, large enough that splitting it in half results
in two datasets that have nearly equivalent statistical properties. In such cases, there may be little need to use k-fold
cross-validation as an evaluation of the algorithm and a train and test split may be just as reliable.