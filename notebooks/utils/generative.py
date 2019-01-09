import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import dirichlet

class Gaussian(object):
    def __init__(self, mean=[0], cov=[1]):
        self.mean = np.array(mean)
        self.cov = np.array(cov)
        self.rv = multivariate_normal(mean=self.mean, cov=self.cov)
        self.n_features = len(mean)

    def pdf(self, x):
        '''Returns pdf value for `x`.'''
        return self.rv.pdf(x)

    def sample(self, size=None, **kwargs):
        return self.rv.rvs(size)

    def __str__(self):
        return 'Gaussian(mean = {}, cov = {})'.format(self.mean, self.cov)

class MixtureDistribution(object):
    def __init__(self, priors, distributions):
        self.priors = np.array(priors)
        self.distributions = distributions

    def sample(self, size=None):
        if size is None:
            size = len(self.priors)
        classes = np.random.multinomial(n=1, pvals=self.priors, size=size)
        samples = np.zeros((size, self.distributions[0].n_features))
        for i, size in enumerate(classes.sum(axis=0)):
            samples[np.where(classes[:,i])[0]] = self.distributions[i].sample(size).reshape(-1, self.distributions[0].n_features)
        return samples, classes

    def posterior(self, x, i=None):
        likelihoods = np.array([d.pdf(x) for d in self.distributions])
        Z = np.dot(self.priors, likelihoods).reshape(-1,1)
        p = np.divide(likelihoods.T*self.priors, Z)
        if i is not None:
            p = p[:,i]
        return p

    def pdf(self, x):
        likelihoods = np.array([d.pdf(x) for d in self.distributions])
        return np.dot(self.priors, likelihoods)

    def likelihoods(self, x):
        return np.array([d.pdf(x) for d in self.distributions]).T

    def __repr__(self):
        string = ''
        for p, d in zip(self.priors, self.distributions):
            string += 'prior = {:.2f}, '.format(p)
            string += 'Distribution = {}'.format(d)
            string += '\n'
        return string

class Dirichlet(object):
    def __init__(self, alphas=[1, 1]):
        self.alphas = np.array(alphas)
        self.rv = dirichlet(self.alphas)
        self.n_features = len(alphas)

    def pdf(self, x):
        '''Returns pdf value for `x`.'''
        return np.array([self.rv.pdf(x_i) for x_i in x])

    def sample(self, size=None, **kwargs):
        return self.rv.rvs(size)

    def __str__(self):
        return 'Dirichlet(alphas = {})'.format(self.alphas)
