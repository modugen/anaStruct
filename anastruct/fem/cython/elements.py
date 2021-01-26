from functools import lru_cache


@lru_cache(32000)
def det_moment(kl, kr, q, x, EI, L):
    """
    See notebook in: anastruct/fem/background/primary_m_v.ipynb

    :param kl: (flt) rotational stiffness left
    :param kr: (flt) rotational stiffness right
    :param q: (flt)
    :param x: (flt) Location of bending moment
    :param EI: (flt)
    :param L: (flt) Length of the beam
    :return: (flt)
    """
    return EI * (-L ** 3 * kl * q * (6 * EI + L * kr) / (
                    12 * EI * (12 * EI ** 2 + 4 * EI * L * kl + 4 * EI * L * kr + L ** 2 * kl * kr)) +
                     L * q * x * (12 * EI ** 2 + 5 * EI * L * kl + 3 * EI * L * kr +
                                  L ** 2 * kl * kr) / (2 * EI * (
                            12 * EI ** 2 + 4 * EI * L * kl + 4 * EI * L * kr + L ** 2 * kl * kr)) - q * x ** 2 / (
                                 2 * EI))


@lru_cache(32000)
def det_shear(kl, kr, q, x, EI, L):
    """
    See notebook in: anastruct/fem/background/primary_m_v.ipynb

    :param kl: (flt) rotational stiffness left
    :param kr: (flt) rotational stiffness right
    :param q: (flt)
    :param x: (flt) Location of bending moment
    :param EI: (flt)
    :param L: (flt) Length of the beam
    :return: (flt)
    """
    return EI * (L * q * (12 * EI ** 2 + 5 * EI * L * kl + 3 * EI * L * kr + L ** 2 * kl * kr) /
                 (2 * EI * (12 * EI ** 2 + 4 * EI * L * kl + 4 * EI * L * kr + L ** 2 * kl * kr)) - q * x / EI)


@lru_cache(32000)
def det_moment_linear(q1, q2, L):
    """
    Return left and right moment for linear distributed load. It's only valid for beams without spring supports.

    :param q1: (flt)
    :param q2: (flt)
    :param L: (flt) Length of the beam
    :return: (flt)
    """
    q_linear = q2-q1
    q_const = q1

    return -((q_const / 12) + (q_linear / 30)) * L ** 2, ((q_const / 12) + (q_linear / 20)) * L ** 2


@lru_cache(32000)
def det_shear_linear(q1, q2, L):
    """
    Return left and right moment for linear distributed load. It's only valid for beams without spring supports.

    :param q1: (flt)
    :param q2: (flt)
    :param L: (flt) Length of the beam
    :return: (flt)
    """

    q_linear = q2 - q1
    q_const = q1

    return ((q_const / 2) + (3 * q_linear / 20)) * L, ((q_const / 2) + (7 * q_linear / 20)) * L
