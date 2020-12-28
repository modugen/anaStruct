import numpy as np
from anastruct import SystemElements


def test_example():
    system = SystemElements()
    system.add_element(location=[[0, 0], [3, 4]], EA=5e9, EI=8000)
    system.add_element(location=[[3, 4], [8, 4]], EA=5e9, EI=4000)
    system.q_load(element_id=2, q=-10)
    system.add_support_hinged(node_id=1)
    system.add_support_fixed(node_id=3)

    sol = np.fromstring(
        """0.00000000e+00   0.00000000e+00   1.30206878e-03   1.99999732e-08
       5.24999402e-08  -2.60416607e-03   0.00000000e+00   0.00000000e+00
       0.00000000e+00""",
        float,
        sep=" ",
    )
    assert np.allclose(system.solve(), sol)