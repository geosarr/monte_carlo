import numpy as np
from utils import standard_cauchy_quantile
from typing import Callable, Union


class MonteCarlo:
    def __str__(self) -> str:
        return "MonteCarlo"

    @staticmethod
    def gen_cauchy_by_inversion(size: int) -> np.array:
        """Draw iid standard Cauchy pseudo variables, i.e with
        density f given by f(x) = [pi*(1+x**2)]^(-1), using the
        inversion method. One can show that the inverse function of the cdf,
        which is the quantile function, is given by q(y) = tan[pi*(y-1/2)],
        for y in ([0,1].
        Args:
            size: number of samples
        Returns:
            the array of samples
        """
        return standard_cauchy_quantile(np.random.uniform(0, 1, size))

    @staticmethod
    def gen_pareto(r: float, a: float, size: int) -> np.array:
        """Draw iid Pareto(r, a) pseudo variables, i.e with
        density f given by f(x) = a*(r^a)/(x^(a+1)) if x>r 0 otherwise.
        One can show that  r / U^(1/a) ~ Pareto(r, a), where U ~ Uniform(0,1)
        Args:
            r: non negative parameter of the Pareto distribution
            a: non negative parameter of the Pareto distribution
            size: number of samples
        Returns:
            the array of samples

        """
        assert r > 0, f"r = {r} is not valid, r should be strictly positive"
        assert a > 0, f"a = {a} is not valid, a should be strictly positive"
        return r / np.random.uniform(0, 1, size) ** (1 / a)

    @staticmethod
    def gen_couple_by_inversion(
        inverse_cdf_y: Callable[[Union[float, int]], Union[float, int]],
        inverse_cdf_x_given_y: Callable[[Union[float, int], Union[float, int]], float],
        size: int,
    ) -> np.ndarray:
        """Draw iid couples (X, Y) of pseudo random variables, when the
        inverse of cdf of Y can be computed, and when the inverse of
        the cdf of X given Y can also be computed.
        Args:
            inverse_cdf_y: the pointwise inverse cumulative distribution function of Y
                (i.e can be computed for a given possible value of Y)
            inverse_cdf_x_given_y: the pointwise inverse cumulative distribution function of X given Y
                (i.e can be computed for a given possible value of X given Y=y) the value of X should be
                the first argument and the value of Y the second argument of this function, subsequent
                parameters should be, for example, put as default values of the function.
            size: the number of samples to generate
        Returns:
            the sample of couples
        """
        Y = np.vectorize(inverse_cdf_y)(np.random.uniform(0, 1, size))
        X = np.vectorize(inverse_cdf_x_given_y)(np.random.uniform(0, 1, size), Y)
        return np.c_[X, Y]


# a = MonteCarlo().gen_couple_by_inversion(lambda u: -np.log(u), lambda u,y: u**(1/y), 1000000)
# print(np.mean(a, axis=0))
