import numpy as np
from anastruct import SystemElements


def test_solve_modal_analysis():
    system = SystemElements()
    l = 10
    n = 50
    subd = l / n
    for i in range(n):
        system.add_element(location=[[subd * i, 0], [subd * (i + 1), 0]], EA=5e8, EI=800, linear_density=10)

    system.add_support_hinged(node_id=1)
    system.add_support_roll(node_id=n + 1)

    natural_frequencies = system.solve_modal_analysis()

    def analytical_natural_frequencies(linear_density, l, EI, n):
        # See Rao, Singiresu S. - Vibration of continuous Systems - John Wiley & Sons (2019) - Section 11.5.1
        return (n ** 2) * (np.pi ** 2) * np.sqrt(EI / (linear_density * l ** 4))

    # retrieve fist four eigenvalues, as error increases for higher eigenvalues
    for i in range(4):
        assert np.isclose(natural_frequencies[i], analytical_natural_frequencies(10, 10, 800, i + 1), rtol=1e-2)
