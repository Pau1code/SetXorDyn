import math
import numpy as np
from dp_accounting import accountant
from dp_accounting import common


class LaplaceMechanism:
  """Transforms a function using the Laplace mechanism.

  If f(x) = (Z[1], Z[2], ..., Z[k]), then returns a function that computes
  (Z'[1], Z'[2], ..., Z'[k]), where
      Z'[i] = Z[i] + Y[i],
      Y[i] ~ Lap(x | delta_f / epsilon),
  and Lap(x | b) is given by the probability density function
      Lap(x | b) = (1 / 2b) exp(-|x| / b).

  """

  def __init__(self, f, delta_f, epsilon, random_state=None):
    """Instantiates a LaplaceMechanism.

    Args:
      f: A function which takes as input a database and which returns as output
        a numpy array.
      delta_f: The sensitivity paramater, e.g., the maximum value by which the
        function can change for two databases that differ by only one row.
      epsilon: Differential privacy parameter.
      random_state:  Optional instance of numpy.random.RandomState that is
        used to seed the random number generator.
    """
    self._func = f
    self._delta_f = delta_f
    self._epsilon = epsilon
    self._random_state = random_state or np.random.RandomState()

  def __call__(self, x):
    z = self._func(x)
    return z + self._random_state.laplace(
        size=z.shape, scale=self._delta_f / self._epsilon)


class GeometricMechanism:
  """Transforms a function using the geometric mechanism.
  """

  def __init__(self, f, delta_f, epsilon, random_state=None):
    """Instantiates a geometric mechanism.

    Args:
      f: A function which takes as input a database and which returns as output
        a numpy array.
      delta_f: The sensitivity paramater, e.g., the maximum value by which the
        function can change for two databases that differ by only one row.
      epsilon: Differential privacy parameter.
      random_state:  Optional instance of numpy.random.RandomState that is
        used to seed the random number generator.
    """
    self._func = f
    self._delta_f = delta_f
    self._epsilon = epsilon
    self._random_state = random_state or np.random.RandomState()

  def __call__(self, x):
    z = self._func(x)
    p_geometric = 1 - math.exp(-self._epsilon / self._delta_f)
    x = self._random_state.geometric(size=z.shape, p=p_geometric)
    y = self._random_state.geometric(size=z.shape, p=p_geometric)
    return z + x - y


class GaussianMechanism:
  """Transforms a function using the gaussian mechanism.

  If f(x) = (Z[1], Z[2], ..., Z[k]), then returns a function that computes
  (Z'[1], Z'[2], ..., Z'[k]), where
      Z'[i] = Z[i] + Y[i],
      Y[i] ~ N(x | sigma),
  and N(x | sigma) is given by the probability density function
      N(x | sigma) = exp(-0.5 x^2 / sigma^2) / (sigma * sqrt(2 * pi))

  See Appendix A of Dwork and Roth.
  """

  def __init__(
    self, f, delta_f, epsilon, delta, num_queries=1, random_state=None):
    """Instantiates a gaussian mechanism.

    Args:
      f: A function which takes as input a database and which returns as output
        a numpy array.
      delta_f: The sensitivity paramater, e.g., the maximum value by which the
        function can change for two databases that differ by only one row.
      epsilon: Differential privacy parameter.
      delta: Differential privacy parameter.
      num_queries: The number of queries for which the mechanism is used. Note
        that the constructed mechanism will be (epsilon, delta)-differentially
        private when answering (no more than) num_queries queries.
      random_state:  Optional instance of numpy.random.RandomState that is
        used to seed the random number generator.
    """
    self._func = f
    self._delta_f = delta_f
    self._sigma = accountant.get_smallest_gaussian_noise(
      common.DifferentialPrivacyParameters(epsilon, delta),
      num_queries, sensitivity=delta_f)
    self._random_state = random_state or np.random.RandomState()

  def __call__(self, x):
    z = self._func(x)
    return z + self._random_state.normal(size=z.shape, scale=self._sigma)

class DiscreteGaussianMechanism:
  """Transforms a function using the discrete gaussian mechanism.

  If f(x) = (Z[1], Z[2], ..., Z[k]), then returns a function that computes
  (Z'[1], Z'[2], ..., Z'[k]), where
      Z'[i] = Z[i] + Y[i],
      Y[i] ~ N_Z(x | sigma),
  and N_Z(x | sigma) is given by the probability mass function defined on the
  integers such that N_Z(x | sigma) is proportional to
  exp(-0.5 x^2 / sigma^2) / (sigma * sqrt(2 * pi)) for all integers x.

  See:

  Clément L. Canonne, Gautam Kamath, Thomas Steinke. "The Discrete Gaussian for
  Differential Privacy" Advances in Neural Information Processing Systems 33
  (NeurIPS 2020).
  """

  def __init__(
    self, f, delta_f, epsilon, delta, num_queries=1, random_state=None):
    """Instantiates a discrete gaussian mechanism.

    Args:
      f: A function which takes as input a database and which returns as output
        a numpy array.
      delta_f: The sensitivity paramater, e.g., the maximum value by which the
        function can change for two databases that differ by only one row.
      epsilon: Differential privacy parameter.
      delta: Differential privacy parameter.
      num_queries: The number of queries for which the mechanism is used. Note
        that the constructed mechanism will be (epsilon, delta)-differentially
        private when answering (no more than) num_queries queries.
      random_state:  Optional instance of numpy.random.RandomState that is
        used to seed the random number generator.
    """
    self._func = f
    self._delta_f = delta_f

    self._sigma = accountant.get_smallest_gaussian_noise(
      common.DifferentialPrivacyParameters(epsilon, delta),
      num_queries, sensitivity=delta_f)
    
    self._random_state = random_state or np.random.RandomState()

  def __call__(self, x):

    def sample_discrete_gaussian(*unused):
      # Use rejection sampling of discrete Laplace distribution (Algorithm 3 in
      # Canonne et al.) to sample a discrete Gaussian random variable.
      t = math.floor(self._sigma) + 1

      while True:
        # Generate discrete laplace with parameter t
        p_geometric = 1 - math.exp(-1/t)
        y1 = self._random_state.geometric(p=p_geometric)
        y2 = self._random_state.geometric(p=p_geometric)
        y = y1 - y2

        sigma_sq = self._sigma**2
        p_bernoulli = math.exp(-(abs(y) - sigma_sq/t)**2 * 0.5 / sigma_sq)
        if self._random_state.binomial(1, p_bernoulli) == 1:
          return y

    z = self._func(x)
    return z + np.fromfunction(
      np.vectorize(sample_discrete_gaussian, otypes=[float]), z.shape)
