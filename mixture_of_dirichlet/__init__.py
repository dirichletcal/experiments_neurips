import numpy

from scipy.stats import dirichlet
from scipy.optimize import minimize

from multiprocessing import Pool, cpu_count

from sklearn.preprocessing import label_binarize


class MixDir:

    def __init__(self):

        self.Alpha = None

        self.pi = None

        self.n_components = None

    def fit(self, S=None, Y=None, n_components=None):

        if S is None:
            raise RuntimeError('No data is provided.')
        else:
            n_samples, n_dimensions = numpy.shape(S)

        if Y is None:
            if n_components is None:
                raise RuntimeError('Number of mixture components is not given.')
            else:
                #print('Unsupervised estimation with EM.')

                self.n_components = n_components

                Alpha, pi, EY = EM_estimation(S, n_samples, n_dimensions, n_components)

                self.Alpha = Alpha

                self.pi = pi

        elif numpy.shape(Y)[0] != n_samples:
                raise RuntimeError('Numbers of instances are not matched between S and Y.')
        else:
            #print('Maximum likelihood estimation.')

            if len(numpy.shape(Y)) == 1:
                Y = label_binarize(Y, numpy.unique(Y))

            n_components = numpy.shape(Y)[1]

            self.n_components = n_components

            Alpha, pi, EY = ML_estimation(S, Y, n_samples, n_dimensions, n_components)

            self.Alpha = Alpha

            self.pi = pi

        return EY

    def predict_proba(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, S=None):

        n_samples = numpy.shape(S)[0]

        EY = numpy.zeros([n_samples, self.n_components])

        for i in range(0, self.n_components):

            EY[:, i] = conditional_likelihood(self.Alpha[i, :], S)

        EY = EY / numpy.repeat(numpy.sum(EY, axis=1).reshape([n_samples, 1]), self.n_components, axis=1)

        return EY


def Bayesian_variational_estimation():

    pass


def EM_estimation(S, n_samples, n_dimensions, n_components, n_iter=10):

    Alpha = numpy.ones([n_components, n_dimensions]) - numpy.random.rand(n_components, n_dimensions)

    pi = numpy.ones([1, n_components])

    EY = numpy.zeros([n_samples, n_components])

    for i in range(0, n_iter):

        for j in range(0, n_components):

            EY[:, j] = conditional_likelihood(Alpha[j, :], S)

        EY = EY * numpy.repeat(pi.reshape([1, 2]), n_samples, axis=0)

        EY = EY / numpy.repeat(numpy.sum(EY, axis=1).reshape([n_samples, 1]), n_components, axis=1)

        pi = numpy.mean(EY, axis=0)

        Alpha = GD_update(S=S, Alpha_old=Alpha, EY=EY)

    return Alpha, pi, EY


def ML_estimation(S, Y, n_samples, n_dimensions, n_components):

    Alpha = numpy.ones([n_components, n_dimensions])

    pi = numpy.mean(Y, axis=0)

    EY = numpy.zeros([n_samples, n_components])

    Alpha = GD_update(S=S, Alpha_old=Alpha, Y=Y)

    for i in range(0, n_components):
        EY[:, i] = conditional_likelihood(Alpha[i, :], S)

    EY = EY / numpy.repeat(numpy.sum(EY, axis=1).reshape([n_samples, 1]), n_components, axis=1)

    return Alpha, pi, EY


def GD_update(S, Alpha_old, Y=None, EY=None):

    n_components, n_dimensions = numpy.shape(Alpha_old)

    Alpha_new = numpy.zeros([n_components, n_dimensions])

    if Y is not None:

        for i in range(0, n_dimensions):

            tmp_idx = (Y[:, i] == 1)

            Alpha_new[i, :] = minimize(sum_conditional_NLL, Alpha_old[i, :], args=(S[tmp_idx, :]),
                                       method='l-bfgs-b', options={'disp': True},
                                       bounds=((1e-128, None), ) * n_dimensions).x

    if EY is not None:

        Alpha_new = minimize(sum_marginal_NLL, Alpha_old.flatten(), args=(S, EY),
                             method='l-bfgs-b', options={'disp': True},
                             bounds=((1e-128, None), ) * n_dimensions * n_components).x.reshape([n_components,
                                                                                                 n_dimensions])

    return Alpha_new


def dirichlet_likelihood(S_alpha):

    S, alpha = S_alpha

    n = numpy.shape(S)[0]

    L = numpy.zeros(n)

    for i in range(0, n):
        # FIXME If the S does not sum to one it raises an Exception
        L[i] = dirichlet.pdf(S[i, :], alpha)

    return L


def conditional_likelihood(alpha, S, n_jobs=-1):

    if n_jobs == -1:
        n_jobs = cpu_count()

    # FIXME should we remove this? If we do not add it in certain cases an
    # exception is raised in function dirichlet.pdf(S) in dirichlet_likelihood
    # ValueError: The input vector 'x' must lie within the normal simplex. but
    # np.sum(x, 0) = 1.001.
    S = (S.T/numpy.sum(S, axis=1)).T

    n = numpy.shape(S)[0]

    edges = numpy.linspace(0, n, n_jobs+1).astype('int')

    S_alpha_list = [(S[edges[i]:edges[i+1], :], alpha) for i in range(0, n_jobs)]

    p = Pool(n_jobs)

    L = p.map(dirichlet_likelihood, S_alpha_list)

    L = numpy.concatenate(L)

    p.close()

    return L


def sum_conditional_NLL(alpha, S):

    NLL = -numpy.log(conditional_likelihood(alpha, S))

    return numpy.sum(NLL)


def sum_marginal_NLL(Alpha, S, EY):

    n_samples, n_components = numpy.shape(EY)

    n_dimensions = numpy.shape(S)[1]

    Alpha = Alpha.reshape([n_components, n_dimensions])

    L = numpy.zeros([n_samples, n_components])

    for i in range(0, n_components):

        L[:, i] = conditional_likelihood(Alpha[i, :], S)

    NLL = -numpy.log(numpy.sum(L * EY, axis=1))

    return numpy.sum(NLL)


