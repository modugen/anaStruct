from anastruct.fem.cython.elements import det_moment_linear, det_shear_linear
from anastruct.fem.elements import det_moment, det_shear
import numpy as np
import scipy as sp
import math
from typing import List, Sequence

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from anastruct.fem.system import SystemElements
    from anastruct.fem.elements import Element


def set_force_vector(
        system: "SystemElements", force_list: List[Tuple[int, int, float]]
):
    """
    :param force_list: list containing tuples with the
    1. number of the node,
    2. the number of the direction (1 = x, 2 = z, 3 = y)
    3. the force
    [(1, 3, 1000)] node=1, direction=3 (y), force=1000
    list may contain multiple tuples
    :return: Vector with forces on the nodes
    """
    assert system.system_force_vector is not None
    for id_, direction, force in force_list:
        system.system_force_vector[(id_ - 1) * 3 + direction - 1] += force

    return system.system_force_vector


def prep_matrix_forces(system: "SystemElements"):
    system.system_force_vector = system.system_force_vector = np.zeros(
        len(system._vertices) * 3
    )
    apply_perpendicular_q_load(system)
    apply_point_load(system)
    apply_moment_load(system)


def apply_moment_load(system: "SystemElements"):
    for node_id, Ty in system.loads_moment.items():
        set_force_vector(system, [(node_id, 3, Ty)])


def apply_point_load(system: "SystemElements"):
    for node_id in system.loads_point:
        Fx, Fz = system.loads_point[node_id]
        # system force vector.
        set_force_vector(
            system,
            [
                (node_id, 1, Fx),
                (node_id, 2, Fz),
            ],
        )


def apply_perpendicular_q_load(system: "SystemElements"):
    for element_id in system.loads_dead_load:
        element = system.element_map[element_id]
        if element.q_load is None and element.dead_load == 0:
            continue

        q_perpendicular = element.all_q_load
        parallel_load = False
        if isinstance(q_perpendicular, (float,int)):
            if not (math.isclose(element.q_load + element.dead_load, q_perpendicular)):
                parallel_load = True
        elif isinstance(q_perpendicular, list):
            if not (math.isclose(element.q_load[0] + element.dead_load, q_perpendicular[0])
                    or
                    math.isclose(element.q_load[1] + element.dead_load, q_perpendicular[1])):
                parallel_load = True
        if parallel_load:
            apply_parallel_q_load(system, element)
        if q_perpendicular == 0:
            continue

        kl = element.constitutive_matrix[1][1] * 1e6
        kr = element.constitutive_matrix[2][2] * 1e6

        if isinstance(q_perpendicular, (float,int)):
            if math.isclose(kl, kr):
                left_moment = det_moment(kl, kr, q_perpendicular, 0, element.EI, element.l)
                right_moment = -left_moment
                rleft = det_shear(kl, kr, q_perpendicular, 0, element.EI, element.l)
                rright = rleft
            else:
                # minus because of systems positive rotation
                left_moment = det_moment(kl, kr, q_perpendicular, 0, element.EI, element.l)
                right_moment = -det_moment(
                    kl, kr, q_perpendicular, element.l, element.EI, element.l
                )
                rleft = det_shear(kl, kr, q_perpendicular, 0, element.EI, element.l)
                rright = -det_shear(
                    kl, kr, q_perpendicular, element.l, element.EI, element.l
                )
        elif isinstance(q_perpendicular, list):
            left_moment, right_moment = det_moment_linear(q_perpendicular[0], q_perpendicular[1], element.l)
            rleft, rright = det_shear_linear(q_perpendicular[0], q_perpendicular[1], element.l)

        rleft_x = rleft * math.sin(element.a1)
        rright_x = rright * math.sin(element.a2)

        rleft_z = rleft * math.cos(element.a1)
        rright_z = rright * math.cos(element.a2)

        if element.type == "truss":
            left_moment = 0
            right_moment = 0

        primary_force = np.array(
            [rleft_x, rleft_z, left_moment, rright_x, rright_z, right_moment]
        )
        element.element_primary_force_vector -= primary_force

        # Set force vector
        assert system.system_force_vector is not None
        system.system_force_vector[
        (element.node_id1 - 1) * 3: (element.node_id1 - 1) * 3 + 3
        ] += primary_force[0:3]
        system.system_force_vector[
        (element.node_id2 - 1) * 3: (element.node_id2 - 1) * 3 + 3
        ] += primary_force[3:]


def apply_parallel_q_load(system: "SystemElements", element: "Element"):
    direction = element.q_direction
    # dead load
    factor_dl = abs(math.sin(element.angle))

    def update(Fx: float, Fz: float):
        element.element_primary_force_vector[0] -= Fx
        element.element_primary_force_vector[1] -= Fz
        element.element_primary_force_vector[3] -= Fx
        element.element_primary_force_vector[4] -= Fz

        set_force_vector(
            system,
            [
                (element.node_1.id, 2, Fz),
                (element.node_2.id, 2, Fz),
                (element.node_1.id, 1, Fx),
                (element.node_2.id, 1, Fx),
            ],
        )

    def update_linear(Fxl: float, Fzl: float, Fxr: float, Fzr: float):
        element.element_primary_force_vector[0] -= Fxl
        element.element_primary_force_vector[1] -= Fzl
        element.element_primary_force_vector[3] -= Fxr
        element.element_primary_force_vector[4] -= Fzr

        set_force_vector(
            system,
            [
                (element.node_1.id, 2, Fzl),
                (element.node_2.id, 2, Fzr),
                (element.node_1.id, 1, Fxl),
                (element.node_2.id, 1, Fxr),
            ],
        )

    if direction == "x":
        factor = abs(math.cos(element.angle))

        if isinstance(element.q_load, (float,int)):

            for q_element in (element.q_load * factor, element.dead_load * factor_dl):
                # q_load working at parallel to the elements x-axis          # set the proper direction
                Fx = -q_element * math.cos(element.angle) * element.l * 0.5
                Fz = (
                        q_element
                        * abs(math.sin(element.angle))
                        * element.l
                        * 0.5
                        * np.sign(math.sin(element.angle))
                )

                update(Fx, Fz)
        elif isinstance(element.q_load, list):
            q1 = element.q_load[0] * factor
            q2 = element.q_load[1] * factor
            q_linear = q2 - q1
            dead_load = element.dead_load * factor_dl
            q = q1 + dead_load
            # q_load working at parallel to the elements x-axis          # set the proper direction
            Fx = -q * math.cos(element.angle) * element.l * 0.5
            Fz = (
                    q
                    * abs(math.sin(element.angle))
                    * element.l
                    * 0.5
                    * np.sign(math.sin(element.angle))
            )
            Fxl = Fx - q_linear * math.cos(element.angle) * element.l * 0.166666666666666666667
            Fxr = Fx - q_linear * math.cos(element.angle) * element.l * 0.33333333333333333333
            Fzl = Fz + (
                    q_linear
                    * abs(math.sin(element.angle))
                    * element.l
                    * 0.1666666666666666666667
                    * np.sign(math.sin(element.angle))
            )
            Fzr = Fz + (
                    q_linear
                    * abs(math.sin(element.angle))
                    * element.l
                    * 0.33333333333333333333
                    * np.sign(math.sin(element.angle))
            )
            update_linear(Fxl, Fzl, Fxr, Fzr)

    else:
        if math.isclose(element.angle, 0):
            # horizontal element cannot have parallel forces due to self weight or q-load in y direction.
            return None
        factor = abs(math.sin(element.angle))
        if isinstance(element.q_load, (float,int)):

            for q_element in (element.q_load * factor, element.dead_load * factor_dl):
                # q_load working at parallel to the elements x-axis          # set the proper direction
                Fx = (
                        q_element
                        * math.cos(element.angle)
                        * element.l
                        * 0.5
                        * -np.sign(math.sin(element.angle))
                )
                Fz = q_element * abs(math.sin(element.angle)) * element.l * 0.5

                update(Fx, Fz)
        elif isinstance(element.q_load, list):
            q1 = element.q_load[0] * factor
            q2 = element.q_load[1] * factor
            q_linear = q2 - q1
            dead_load = element.dead_load * factor_dl
            q = q1 + dead_load
            # q_load working at parallel to the elements x-axis          # set the proper direction
            Fx = (
                    q
                    * math.cos(element.angle)
                    * element.l
                    * 0.5
                    * -np.sign(math.sin(element.angle))
            )
            Fz = q * abs(math.sin(element.angle)) * element.l * 0.5

            Fxl = Fx + (
                    q_linear
                    * math.cos(element.angle)
                    * element.l
                    * 0.1666666666666666667
                    * -np.sign(math.sin(element.angle))
            )
            Fxr = Fx + (
                    q_linear
                    * math.cos(element.angle)
                    * element.l
                    * 0.33333333333333333333
                    * -np.sign(math.sin(element.angle))
            )
            Fzl = Fz + q_linear * abs(math.sin(element.angle)) * element.l * 0.1666666666666666667

            Fzr = Fz + q_linear * abs(math.sin(element.angle)) * element.l * 0.33333333333333333333

            update_linear(Fxl, Fzl, Fxr, Fzr)




def dead_load(system: "SystemElements", g: float, element_id: int):
    system.loads_dead_load.add(element_id)
    system.element_map[element_id].dead_load = g


def assemble_system_matrix(
        system: "SystemElements", validate: bool = False, geometric_matrix: bool = False
):
    """
    Shape of the matrix = n nodes * n d.o.f.
    Shape = n * 3
    """
    system._remainder_indexes = []
    if not geometric_matrix:
        shape = len(system.node_map) * 3
        system.shape_system_matrix = shape
        system.system_matrix = np.zeros((shape, shape))

    assert system.system_matrix is not None
    for matrix_index, K in system.system_spring_map.items():
        #  first index is row, second is column
        system.system_matrix[matrix_index][matrix_index] += K

    # Determine the elements location in the stiffness matrix.
    # system matrix [K]
    #
    # [fx 1] [K         |  \ node 1 starts at row 1
    # |fz 1] |  K       |  /
    # |Ty 1] |   K      | /
    # |fx 2] |    K     |  \ node 2 starts at row 4
    # |fz 2] |     K    |  /
    # |Ty 2] |      K   | /
    # |fx 3] |       K  |  \ node 3 starts at row 7
    # |fz 3] |        K |  /
    # [Ty 3] [         K] /
    #
    #         n   n  n
    #         o   o  o
    #         d   d  d
    #         e   e  e
    #         1   2  3
    #
    # thus with appending numbers in the system matrix: column = row

    for i in range(len(system.element_map)):
        element = system.element_map[i + 1]
        element_matrix = element.stiffness_matrix

        # n1 and n2 are starting indexes of the rows and the columns for node 1 and node 2
        n1 = (element.node_1.id - 1) * 3
        n2 = (element.node_2.id - 1) * 3
        system.system_matrix[n1: n1 + 3, n1: n1 + 3] += element_matrix[0:3, :3]
        system.system_matrix[n1: n1 + 3, n2: n2 + 3] += element_matrix[0:3, 3:]

        system.system_matrix[n2: n2 + 3, n1: n1 + 3] += element_matrix[3:6, :3]
        system.system_matrix[n2: n2 + 3, n2: n2 + 3] += element_matrix[3:6, 3:]

    # returns True if symmetrical.
    if validate:
        assert np.allclose((system.system_matrix.transpose()), system.system_matrix)

def assemble_system_mass_matrix(
        system: "SystemElements"
):
    """
    Shape of the matrix = n nodes * n d.o.f.
    Shape = n * 3
    """
    system._remainder_indexes = []
    shape = len(system.node_map) * 3
    system.shape_system_matrix = shape
    system.system_mass_matrix = np.zeros((shape, shape))


    # Determine the elements location in the modal matrix.
    # system matrix [K]
    #
    # [mfx 1] [K         |  \ node 1 starts at row 1
    # |mfz 1] |  K       |  /
    # |mTy 1] |   K      | /
    # |mfx 2] |    K     |  \ node 2 starts at row 4
    # |mfz 2] |     K    |  /
    # |mTy 2] |      K   | /
    # |mfx 3] |       K  |  \ node 3 starts at row 7
    # |mfz 3] |        K |  /
    # [mTy 3] [         K] /
    #
    #         n   n  n
    #         o   o  o
    #         d   d  d
    #         e   e  e
    #         1   2  3
    #
    # thus with appending numbers in the system matrix: column = row

    for i in range(len(system.element_map)):
        element = system.element_map[i + 1]
        element_matrix = element.mass_matrix

        # n1 and n2 are starting indexes of the rows and the columns for node 1 and node 2
        n1 = (element.node_1.id - 1) * 3
        n2 = (element.node_2.id - 1) * 3
        system.system_mass_matrix[n1: n1 + 3, n1: n1 + 3] += element_matrix[0:3, :3]
        system.system_mass_matrix[n1: n1 + 3, n2: n2 + 3] += element_matrix[0:3, 3:]

        system.system_mass_matrix[n2: n2 + 3, n1: n1 + 3] += element_matrix[3:6, :3]
        system.system_mass_matrix[n2: n2 + 3, n2: n2 + 3] += element_matrix[3:6, 3:]

    # returns True if symmetrical.
    assert np.allclose((system.system_matrix.transpose()), system.system_matrix)

def set_displacement_vector(system, nodes_list):
    """
    :param nodes_list: list containing tuples with
    1.the node
    2. the d.o.f. that is set
    :return: Vector with the displacements of the nodes (If displacement is not known, the value is set
    to NaN)
    """
    if system.system_displacement_vector is None:
        system.system_displacement_vector = np.ones(len(system._vertices) * 3) * np.NaN

    for i in nodes_list:
        index = (i[0] - 1) * 3 + i[1] - 1

        try:
            system.system_displacement_vector[index] = 0
        except IndexError as e:
            raise IndexError(
                e,
                "This often occurs if you set supports before the all the elements are modelled. "
                "First finish the model.",
            )
    return system.system_displacement_vector


def process_conditions(system):
    indexes = []
    # remove the unsolvable values from the matrix and vectors
    for i in range(system.shape_system_matrix):
        if system.system_displacement_vector[i] == 0:
            indexes.append(i)
        else:
            system._remainder_indexes.append(i)

    system.system_displacement_vector = np.delete(
        system.system_displacement_vector, indexes, 0
    )
    if system.system_force_vector is not None:
        system.reduced_force_vector = np.delete(system.system_force_vector, indexes, 0)
    system.reduced_system_matrix = np.delete(system.system_matrix, indexes, 0)
    system.reduced_system_matrix = np.delete(system.reduced_system_matrix, indexes, 1)
    if system.system_mass_matrix is not None:
        system.reduced_system_mass_matrix = np.delete(system.system_mass_matrix, indexes, 0)
        system.reduced_system_mass_matrix = np.delete(system.reduced_system_mass_matrix, indexes, 1)


def process_supports(system):
    for node in system.supports_hinged:
        set_displacement_vector(system, [(node.id, 1), (node.id, 2)])

    for i in range(len(system.supports_roll)):
        if not system.supports_roll_rotate[i]:
            set_displacement_vector(
                system,
                [
                    (system.supports_roll[i].id, system.supports_roll_direction[i]),
                    (system.supports_roll[i].id, 3),
                ],
            )
        else:
            set_displacement_vector(
                system,
                [(system.supports_roll[i].id, system.supports_roll_direction[i])],
            )

    for node in system.supports_rotational:
        set_displacement_vector(system, [(node.id, 3)])

    for node in system.supports_fixed:
        set_displacement_vector(system, [(node.id, 1), (node.id, 2), (node.id, 3)])

    for node, roll in system.supports_spring_x:
        if not roll:
            set_displacement_vector(system, [(node.id, 2)])

    for node, roll in system.supports_spring_z:
        if not roll:
            set_displacement_vector(system, [(node.id, 1)])

    for node, roll in system.supports_spring_y:
        if not roll:
            set_displacement_vector(system, [(node.id, 1), (node.id, 2)])

    for node_id, angle in system.inclined_roll.items():
        for el in system.node_element_map[node_id]:
            if el.node_1.id == node_id:
                el.a1 = el.angle + angle
            elif el.node_2.id == node_id:
                el.a2 = el.angle + angle

            el.compile_kinematic_matrix(el.a1, el.a2, el.l)
            el.compile_stiffness_matrix()
