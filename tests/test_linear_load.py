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

    assert np.isclose(np.max(abs(system.element_map[1].shear_force)), 10.87, rtol=1e-2)
    assert np.isclose(np.max(abs(system.element_map[2].shear_force)), 4.17, rtol=1e-2)
    assert np.isclose(np.max(abs(system.element_map[1].bending_moment)), 4.62, rtol=1e-2)
    assert np.isclose(np.max(abs(system.system_displacement_vector)), 3.7673e-3, rtol=1e-2)


def test_moment_q_load():
    system = SystemElements()
    start=0
    step = 0.1
    for i in range(20):
        system.add_element(location=[[start, 0], [start+step, 0]], EA=5e2, EI=800)
        start+=step
        system.q_moment(element_id=1+i, Ty=10)


    system.add_support_hinged(node_id=1)
    system.add_support_roll(node_id=21)

    system.solve()

    assert np.isclose(system.reaction_forces[1].Fz, 10)

def test_moment_load_benchmark():
    system = SystemElements()
    system.add_element(location=[[3.75, 0]], EA=5e12, EI=5e12)
    system.add_element(location=[[7.5,0]], EA=5e12, EI=5e12)

    system.q_moment(element_id=1, Ty=1.91)
    system.q_moment(element_id=2, Ty=1.91)
    system.q_load(element_id=1, q=3.62, direction="y")
    system.q_load(element_id=2, q=3.62, direction="y")

    system.add_support_spring(1, 2, 5)
    system.add_support_spring(2, 2, 2.5)
    system.add_support_spring(3, 2, 2.5)


    system.solve()
    assert np.isclose(np.max(abs(system.element_map[1].bending_moment)), 13.89,rtol=1e-2)

    assert np.isclose(system.reaction_forces[1].Fz, 11.93,rtol=1e-2)

def test_moment_load_benchmark_2():
    system = SystemElements()
    system.add_element(location=[[3.125, 0]], EA=5e12, EI=5e12)
    system.add_element(location=[[5.625+3.125,0]], EA=5e12, EI=5e12)

    system.q_moment(element_id=1, Ty=1.836)
    system.q_moment(element_id=2, Ty=1.836)
    system.q_load(element_id=1, q=4.08, direction="y")
    system.q_load(element_id=2, q=4.08, direction="y")

    system.add_support_spring(1, 2, 3.75)
    system.add_support_spring(2, 2, 2.5)
    system.add_support_spring(3, 2, 2.5)

    #system.add_support_hinged(1)
    #system.add_support_hinged(2)
    #system.add_support_hinged(3)

    system.solve()
    assert np.isclose(np.max(abs(system.element_map[2].bending_moment)), 24.87,rtol=1e-2)
    assert np.isclose(np.max(abs(system.element_map[1].shear_force)), 13.2,rtol=1e-2)

def test_linear_parallel_q_load():
    system = SystemElements()
    system.add_element(location=[[0, 0], [1, 0]], EA=5e2, EI=800)
    system.add_element(location=[[1, 0], [2, 0]], EA=5e2, EI=800)
    system.q_load(element_id=1, q=10, q2=20, direction="x")
    system.add_support_hinged(node_id=1)
    system.add_support_roll(node_id=3)

    system.solve()

    assert np.isclose(system.reaction_forces[1].Fx, -15)
    assert np.isclose(np.max(system.system_displacement_vector), 0.01666667)
