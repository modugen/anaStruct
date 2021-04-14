from anastruct import SystemElements
import numpy as np


def test_timoshenko_discrete():
    system = SystemElements()
    system.add_element(location=[[0.25, 0]], EA=1.4e8, EI=1.167e5, GA=8.75e6)
    system.add_element(location=[[0.5, 0]], EA=1.4e8, EI=1.167e5, GA=8.75e6)
    system.add_element(location=[[0.75, 0]], EA=1.4e8, EI=1.167e5, GA=8.75e6)
    system.add_element(location=[[1, 0]], EA=1.4e8, EI=1.167e5, GA=8.75e6)
    system.q_load(element_id=1, q=5000, direction="y")
    system.q_load(element_id=2, q=5000, direction="y")
    system.q_load(element_id=3, q=5000, direction="y")
    system.q_load(element_id=4, q=5000, direction="y")

    system.add_support_hinged(node_id=1)
    system.add_support_roll(node_id=5)

    system.solve()

    assert np.isclose(abs(system.system_displacement_vector[7]), 6.44e-4, rtol=1e-2)

def test_timoshenko_continuous():
    system = SystemElements()
    system.add_element(location=[[0, 0], [1, 0]], EA=1.4e8, EI=1.167e5, GA=8.75e6)
    system.q_load(element_id=1, q=5000, direction="y")

    system.add_support_hinged(node_id=1)
    system.add_support_roll(node_id=2)

    system.solve()

    assert np.isclose(np.max(abs(system.element_map[1].deflection)), 6.44e-4, rtol=1e-2)

def test_bernoulli():
    system = SystemElements()
    system.add_element(location=[[0, 0], [1, 0]], EA=1.4e8, EI=1.167e5)
    system.q_load(element_id=1, q=5000, direction="y")

    system.add_support_hinged(node_id=1)
    system.add_support_roll(node_id=2)

    system.solve()

    assert np.isclose(np.max(abs(system.element_map[1].deflection)), 5.58e-4, rtol=1e-2)
