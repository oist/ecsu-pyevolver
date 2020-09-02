import re
import numpy as np
import os


def is_int(s):
    return re.match(r'^\d+$', s) is not None


def linmap(vin, rin, rout):
    """
    Map a vector between 2 ranges.
    :param vin: input vector to be mapped
    :param rin: range of vin to map from
    :param rout: range to map to
    :return: mapped output vector
    :rtype np.ndarray
    """
    a = rin[0]
    b = rin[1]
    c = rout[0]
    d = rout[1]
    return ((c + d) + (d - c) * ((2 * vin - (a + b)) / (b - a))) / 2


# TODO: consider replacing 0 values with small positive number to avoid slashing everything to 0
def harmonic_mean(fit_list):
    if 0 in fit_list:
        return 0
    else:
        return len(fit_list) / np.sum(1.0 / np.array(fit_list))


def rolling_mean(prev, current, counter):
    return (prev * (counter - 1) + current) / counter


def add_noise(vector, random_state, variance=0.05):
    magnitude = random_state.normal(0, variance)
    return vector + magnitude


def vector_norm(vector):
    return np.sqrt(vector.dot(vector))


def make_rand_vector(dims, random_state):
    """
    Generate a random unit vector.  This works by first generating a vector each of whose elements
    is a random Gaussian and then normalizing the resulting vector.
    """
    vec = random_state.normal(0, 1, dims)
    # mag = sum(vec ** 2) ** .5
    mag = vector_norm(vec)
    return vec / mag


def linear_scaling(min_value, max_value, avg_value, multiplier):
    # Compute the coefficients for linear fitness scaling, see Goldberg, pp. 76-79
    if min_value > (multiplier * avg_value - max_value) / (multiplier - 1):
        delta = max_value - avg_value
        if delta > 0:
            return (multiplier - 1) * avg_value / delta
        else:
            return 0
    else:
        delta = avg_value - min_value
        if delta > 0:
            return avg_value / delta
        else:
            return 0


def make_dir_if_not_exists(dir_path):
    if os.path.exists(dir_path):
        assert os.path.isdir(dir_path), 'Path {} is not a directory'.format(dir_path)
        return
    os.makedirs(dir_path)


if __name__ == "__main__":
    print(is_int('3'))