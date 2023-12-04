from scipy.stats import truncnorm


def truncated_normal(center, low, high, size=1):
    sigma = (high - low) / 6
    a, b = (low - center) / sigma, (high - center) / sigma
    return truncnorm.rvs(a, b, loc=center, scale=sigma, size=size)[0]


def truncated_normal_int(center, low, high, size=1):
    return int(truncated_normal(center, low, high, size))