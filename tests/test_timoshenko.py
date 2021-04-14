from anastruct import SystemElements
import numpy as np


def test_timoshenko():
    system = SystemElements()
    system.add_element(location=[[0, 0], [1, 0]], EA=1.4e8, EI=1.4e4,GA=8.75e6)
    system.q_load(element_id=1, q=5, direction="y")

    system.add_support_hinged(node_id=1)
    system.add_support_roll(node_id=2)

    system.solve()

    assert np.isclose(np.max(abs(system.element_map[1].element_displacement_vector)), 6.44e-4, rtol=10e3)

def test_bernoulli():
    system = SystemElements()
    system.add_element(location=[[0, 0], [1, 0]], EA=1.4e8, EI=1.4e4)
    system.q_load(element_id=1, q=5, direction="y")

    system.add_support_hinged(node_id=1)
    system.add_support_roll(node_id=2)

    system.solve()

    assert np.isclose(np.max(abs(system.element_map[1].element_displacement_vector)), 5.58e-4, rtol=10e3)
