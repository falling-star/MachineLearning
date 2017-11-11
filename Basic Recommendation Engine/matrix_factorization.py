import numpy


#    R     : A Matrix to be factorized, dimension N x M
#    P     : An initial matrix of dimension N x K
#    Q     : An initial matrix of dimension M x K
#    K     : The number of latent features (Hidden features which contribute to ratings)
#    steps : The maximum number of steps to perform the optimisation
#    alpha : The learning rate (for Gradient Descent)
#    reg_lambda  : The regularization parameter ( for GD Approach)


def matrix_factorization_gradient_descent(R, P, Q, K, steps=5000, alpha=0.0002, reg_lambda=0.02):
    Q = Q.T
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    # calculate error for (i, j) rating
                    eij = R[i][j] - numpy.dot(P[i, :], Q[:, j])
                    for k in xrange(K):
                        # Update P & Q GD parameters by their respective derivative slope *  alpha learning rate
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - reg_lambda * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - reg_lambda * Q[k][j])
        eR = numpy.dot(P, Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    # calculate total error for rated movies vs user
                    e = e + pow(R[i][j] - numpy.dot(P[i, :], Q[:, j]), 2)
                    for k in xrange(K):
                        # add regularized cost to the error function
                        e = e + (reg_lambda / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        if e < 0.001:
            break
    return P, Q.T


R = [
        [2, 1, 0, 4],
        [5, 0, 5, 5],
        [2, 1, 0, 5],
        [2, 2, 0, 3],
        [5, 0, 5, 4],
    ]

R = numpy.array(R)

N = len(R)
M = len(R[0])
K = 2

P = numpy.random.rand(N, K)
Q = numpy.random.rand(M, K)

nP, nQ = matrix_factorization_gradient_descent(R, P, Q, K)
nQ = nQ.T

print (nP)

print (nQ)

mx = numpy.matrix(nP)
my = numpy.matrix(nQ)

print (mx * my)

'''

Output Matrix after Factorization - R as  (P * Q^T)

[[ 2.00481913  1.21639338  3.09708768  3.90139526]
 [ 4.80130429  2.08024915  5.16000299  4.97268022]
 [ 2.03988562  1.39602931  3.5804297   4.80067018]
 [ 2.11971995  1.11555361  2.8123709   3.22996568]
 [ 5.06300413  1.96932643  4.83335566  4.06660168]]

'''