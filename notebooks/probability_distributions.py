from math import gamma
from operator import mul
import numpy as np
from functools import reduce

class Dirichlet(object):
    def __init__(self, alpha):
        self._alpha = np.array(alpha)
        self._coef = gamma(np.sum(self._alpha)) / \
                     reduce(mul, [gamma(a) for a in self._alpha])
    def pdf(self, x):
        '''Returns pdf value for `x`.'''
        return self._coef * reduce(mul, [xx ** (aa - 1)
                                         for (xx, aa)in zip(x, self._alpha)])

    def sample(self, size=None, **kwargs):
        return np.random.dirichlet(self._alpha, size=size, **kwargs)

    def __str__(self):
        return np.array2string(self._alpha, separator=',', precision=2)

    def __repr__(self):
        return self._alpha.__repr__()


class MixtureDistribution(object):
    def __init__(self, priors, distributions):
        self.priors = np.array(priors)
        self.distributions = distributions

    def sample(self, size=None):
        if size is None:
            size = len(self.priors)
        classes = np.random.multinomial(n=1, pvals=self.priors, size=size)
        samples = np.empty_like(classes, dtype='float')
        for i, size in enumerate(classes.sum(axis=0)):
            samples[np.where(classes[:,i])[0]] = self.distributions[i].sample(size)
        return samples, classes

    def posterior(self, pvalues, c=0):
        likelihoods = np.array([d.pdf(pvalues) for d in self.distributions])
        Z = np.dot(likelihoods, self.priors)
        return np.divide(likelihoods[c]*self.priors[c], Z)

    def pdf(self, pvalues):
        likelihoods = np.array([d.pdf(pvalues) for d in self.distributions])
        return np.dot(self.priors, likelihoods)

    def __repr__(self):
        string = ''
        for p, d in zip(self.priors, self.distributions):
            string += 'prior = {}, '.format(p)
            string += 'Distribution = {}'.format(d)
            string += '\n'
        return string
