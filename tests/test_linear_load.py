import numpy as np
from anastruct import SystemElements

def test_linear_q_load():
    system = SystemElements()
    system.add_element(location=[[0, 0], [1, 0]], EA=5e8, EI=800)
    system.add_element(location=[[1, 0], [2, 0]], EA=5e8, EI=800)
    system.q_load(element_id=1, q=-10, q2=-20, direction="y")
    system.add_support_hinged(node_id=1)
    system.add_support_roll(node_id=3)

    system.solve()

    assert np.isclose(np.max(abs(system.element_map[1].shear_force)), 10.87, rtol=10e3)
    assert np.isclose(np.max(abs(system.element_map[2].shear_force)), 4.17, rtol=10e3)
    assert np.isclose(np.max(abs(system.element_map[1].bending_moment)), 4.62, rtol=10e3)
    assert np.isclose(np.max(abs(system.system_displacement_vector)), 2.135e-3, rtol=10e3)

def test_linear_parallel_q_load():
    system = SystemElements()
    system.add_element(location=[[0, 0], [1, 0]], EA=5e2, EI=800)
    system.add_element(location=[[1, 0], [2, 0]], EA=5e2, EI=800)
    system.q_load(element_id=1, q=10, q2=20, direction="x")
    system.add_support_hinged(node_id=1)
    system.add_support_roll(node_id=3)

    system.solve()

    assert np.isclose(system.reaction_forces[1].Fx,-15)
    assert np.isclose(np.max(system.system_displacement_vector), 0.01666667)