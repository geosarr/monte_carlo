from typing import Callable, Union, Any
import numpy as np
from utils import cauchy_quantile

class MonteCarlo:
    def __str__(self) -> str:
        return "MonteCarlo"

    @staticmethod
    def gen_rand_cauchy_by_inversion(theta: int, size: int) -> np.array:
        """Generates iid Cauchy(theta) pseudo variables, i.e with
        density f given by f(x) = theta * [pi*(theta**2 + x**2)]^(-1), using the
        inversion method. One can show that the inverse function of the cdf,
        which is the quantile function, is given by q(y) = tan[pi*(y-1/2)],
        for y in ([0,1].
        Args:
            size: number of samples to generate
        Returns:
            the array of samples
        """
        assert (
            theta > 0
        ), f"theta = {theta} is not valid, theta should be strictly positive"
        return cauchy_quantile(np.random.uniform(0, 1, size), theta)

    @staticmethod
    def gen_rand_pareto(scale: float, shape: float, size: int) -> np.array:
        """Generates iid Pareto(r, a) pseudo variables, i.e with
        density f given by f(x) = shape*(scale^shape)/(x^(shape+1)) if x>r 0 otherwise.
        One can show that  scale / U^(1/shape) ~ Pareto(scale, shape), where U ~ Uniform(0,1)
        Args:
            scale: non negative parameter of the Pareto distribution
            shape: non negative parameter of the Pareto distribution
            size: number of samples to generate
        Returns:
            the array of samples
        """
        assert scale > 0, f"scale = {scale} is not valid, scale should be strictly positive"
        assert shape > 0, f"shape = {shape} is not valid, shape should be strictly positive"
        return scale / np.random.uniform(0, 1, size) ** (1 / shape)

    @classmethod
    def gen_rand_normal_by_accept_reject(
        cls, mean: float, std: float, size: int
    ) -> np.array:
        """Generates iid pseudo random normal variables with a given mean and
        a given standard deviation using accept-reject method and a Laplace
        distribution as a proposal
        Args:
            mean: mean of the distribution
            std: the standard deviation of the distribution
            size: the number of samples
        Returns:
            the samples from a normal distribution
        """
        assert std > 0, f"std = {std} is not valid, std should be strictly positive"
        standard_normal = cls.gen_rand_by_accept_reject(
            gen_from_proposal=lambda size: np.random.laplace(
                loc=0, scale=1 / 0.24, size=size
            ),
            ratio_func=lambda x: (np.sqrt(2 / np.pi) / 0.24)
            * np.exp(-(x**2) / 2 - 0.24 * np.absolute(x)),
            avg_acc=1,
            size=size,
        )
        return mean + standard_normal * std

    @classmethod
    def gen_rand_by_accept_reject(
        cls,
        gen_from_proposal: Callable[[int, Any], np.array],
        ratio_func: Callable[[float], float],
        avg_acc: int,
        size: int,
    ) -> np.array:
        """Draw iid pseudo variables by accept reject when the density
        functions can be computed pointwise and it is possible to simulate
        from the proposal distribution.
        Args:
            gen_from_proposal: function used to generate iid pseudo variales
            from the proposal distribution, its first argument should be the size
            of the sample argument
            avg_acc: is the integer part of the average of acceptance stopping
                time tau = inf{i in N*, a proposal is accepted} ~ Geometric(1/m),
                here, avg_acc = int(m).
            size: number of samples
        Returns:
            the samples of the target distribution
        """
        results = np.array([])
        nb_prop = size * avg_acc  # number of proposal to simulate at each iteration
        counter = 0
        while counter < size:
            # generate the proposals and the uniforms
            proposals = gen_from_proposal(nb_prop)
            uniforms = np.random.uniform(0, 1, nb_prop)

            # get the accepted proposals, if any
            id_acc_prop = np.where(uniforms <= ratio_func(proposals))[0]

            # add the accepted ones, if any
            results = np.concatenate([results, proposals[id_acc_prop]])
            counter += len(id_acc_prop)

        return results[:size]

    @staticmethod
    def gen_rand_couple_by_inversion(
        inverse_cdf_y: Callable[[Union[float, int]], Union[float, int]],
        inverse_cdf_x_given_y: Callable[[Union[float, int], Union[float, int]], float],
        size: int,
    ) -> np.ndarray:
        """Generates iid couples (X, Y) of pseudo random variables, when the
        inverse of cdf of Y can be computed, and when the inverse of
        the cdf of X given Y can also be computed.
        Args:
            inverse_cdf_y: the pointwise inverse cumulative distribution function of Y
                (i.e can be computed for a given possible value of Y)
            inverse_cdf_x_given_y: the pointwise inverse cumulative distribution function of
                X given Y (i.e can be computed for a given possible value of X given Y=y) the
                value of X should be the first argument and the value of Y the second argument
                of this function, subsequent parameters should be, for example, put as default
                values of the function.
            size: the number of samples to generate
        Returns:
            the sample of couples
        """
        var_y = np.vectorize(inverse_cdf_y)(np.random.uniform(0, 1, size))
        var_x = np.vectorize(inverse_cdf_x_given_y)(np.random.uniform(0, 1, size), var_y)
        return np.c_[var_x, var_y]
