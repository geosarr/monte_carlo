from unittest import TestCase
from functools import partial
from scipy import stats
from monte_carlo import MonteCarlo


class TestGenerators(TestCase):
    def setUp(self):
        self.size = int(1e6)

    def test_gen_rand_cauchy_by_inversion(self):
        with self.assertRaises(Exception) as _:
            MonteCarlo().gen_rand_cauchy_by_inversion(theta=-1.0, size=self.size)
        cauchy = MonteCarlo().gen_rand_cauchy_by_inversion(theta=1.0, size=self.size)
        self.assertTrue(len(cauchy) == self.size)
        kolgomorov_smirnov_test = stats.kstest(cauchy, stats.cauchy.cdf)
        self.assertTrue(
            kolgomorov_smirnov_test.pvalue >= 0.05
        )  # do not reject null hypothesis at level 0.05

    def test_gen_rand_pareto(self):
        with self.assertRaises(Exception) as _:
            MonteCarlo().gen_rand_pareto(scale=-1.0, shape=2.0, size=self.size)
        with self.assertRaises(Exception) as _:
            MonteCarlo().gen_rand_pareto(scale=1.0, shape=-2.0, size=self.size)
        _a = 1.0
        pareto = MonteCarlo().gen_rand_pareto(scale=1.0, shape=_a, size=self.size)
        self.assertTrue(len(pareto) == self.size)
        kolgomorov_smirnov_test = stats.kstest(pareto, partial(stats.pareto.cdf, b=_a))
        self.assertTrue(
            kolgomorov_smirnov_test.pvalue >= 0.05
        )  # do not reject null hypothesis at level 0.05

    def test_gen_rand_normal_by_accept_reject(self):
        with self.assertRaises(Exception) as _:
            MonteCarlo().gen_rand_normal_by_accept_reject(
                mean=0.0, std=-1.0, size=self.size
            )
        gaussian = MonteCarlo().gen_rand_normal_by_accept_reject(
            mean=0.0, std=1.0, size=self.size
        )
        self.assertTrue(len(gaussian) == self.size)
        kolgomorov_smirnov_test = stats.kstest(gaussian, stats.norm.cdf)
        # print(kolgomorov_smirnov_test.pvalue, gaussian.mean(), gaussian.std())
        self.assertTrue(
            kolgomorov_smirnov_test.pvalue >= 0.05
        )  # do not reject null hypothesis at level 0.05


# a = MonteCarlo().gen_couple_by_inversion(lambda u: -np.log(u), lambda u,y: u**(1/y), 1000000)
# print(np.mean(a, axis=0))
