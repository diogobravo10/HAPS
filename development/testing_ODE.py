import importlib

import numpy as np
import openmdao.api as om
import lateral_kinematics as lateralkinematics
import kinematics3D as kinematics_3d
import longitudinal_kinematics as longitudinalkinematics


def test_lateral_kinematics():
    num_nodes = 5

    p = om.Problem(model=om.Group())

    ivc = p.model.add_subsystem('vars', om.IndepVarComp())
    ivc.add_output('V', shape=(num_nodes,), units='m/s')
    ivc.add_output('phi', shape=(num_nodes,), units='deg')
    ivc.add_output('psi', shape=(num_nodes,), units='deg')
    ivc.add_output('gg', shape=(num_nodes,), units='deg')

    p.model.add_subsystem('ode', lateralkinematics.Kinematics(num_nodes=num_nodes))

    p.model.connect('vars.V', 'ode.V')
    p.model.connect('vars.phi', 'ode.phi')
    p.model.connect('vars.psi', 'ode.psi')
    p.model.connect('vars.gg', 'ode.gg')

    p.setup(force_alloc_complex=True)

    p.set_val('vars.V', 10*np.random.random(num_nodes))
    p.set_val('vars.phi', 10*np.random.uniform(1, 179, num_nodes))
    p.set_val('vars.psi', np.random.uniform(0, 360, num_nodes))
    p.set_val('vars.gg', np.random.uniform(-10, 10, num_nodes))

    p.run_model()
    return p.check_partials(method='cs', compact_print=True)


def test_3d_kinematics():
    num_nodes = 5

    p = om.Problem(model=om.Group())

    ivc = p.model.add_subsystem('vars', om.IndepVarComp())
    ivc.add_output('psi', shape=(num_nodes,), units='rad')
    ivc.add_output('V', shape=(num_nodes,), units='m/s')
    ivc.add_output('gg', shape=(num_nodes,), units='rad')
    ivc.add_output('h', shape=(num_nodes,), units='m')
    ivc.add_output('phi', shape=(num_nodes,), units='rad')
    ivc.add_output('aa', shape=(num_nodes,), units='rad')
    ivc.add_output('Throttle', shape=(num_nodes,))

    p.model.add_subsystem('ode', kinematics_3d.Kinematics3D(num_nodes=num_nodes))

    for name in ['psi', 'V', 'gg', 'h', 'phi', 'aa', 'Throttle']:
        p.model.connect(f'vars.{name}', f'ode.{name}')

    p.setup(force_alloc_complex=True)

    # V kept well away from 0 (1/V terms in the dynamics), gg/phi kept away from
    # +-90 deg (cos_gg divisions), h kept within the standard atmosphere's range
    p.set_val('vars.psi', np.random.uniform(0, 2*np.pi, num_nodes))
    p.set_val('vars.V', np.random.uniform(5, 15, num_nodes))
    p.set_val('vars.gg', np.radians(np.random.uniform(-10, 10, num_nodes)))
    p.set_val('vars.h', np.random.uniform(15000, 24000, num_nodes))
    p.set_val('vars.phi', np.radians(np.random.uniform(-20, 20, num_nodes)))
    p.set_val('vars.aa', np.radians(np.random.uniform(0, 10, num_nodes)))
    p.set_val('vars.Throttle', np.random.uniform(0.2, 0.8, num_nodes))

    p.run_model()
    return p.check_partials(method='fd', form='central', step=1e-6, compact_print=True)


def test_longitudinal_kinematics():
    num_nodes = 5

    p = om.Problem(model=om.Group())

    ivc = p.model.add_subsystem('vars', om.IndepVarComp())
    ivc.add_output('V', shape=(num_nodes,), units='m/s')
    ivc.add_output('gg', shape=(num_nodes,), units='deg')

    p.model.add_subsystem('ode', longitudinalkinematics.Kinematics(num_nodes=num_nodes))

    p.model.connect('vars.V', 'ode.V')
    p.model.connect('vars.gg', 'ode.gg')

    p.setup(force_alloc_complex=True)

    p.set_val('vars.V', 10*np.random.random(num_nodes))
    p.set_val('vars.gg', np.random.uniform(-10, 10, num_nodes))

    p.run_model()
    return p.check_partials(method='cs', compact_print=True)


if __name__ == '__main__':
    # test_lateral_kinematics()
    # test_3d_kinematics()
    test_longitudinal_kinematics()
