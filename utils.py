from numpy import tan, pi

def cauchy_quantile(number: float, theta: float) -> float:
    assert (
            theta > 0
    ), f"theta = {theta} is not valid, theta should be strictly positive"
    return tan(pi * number - 1 / 2) * theta
    