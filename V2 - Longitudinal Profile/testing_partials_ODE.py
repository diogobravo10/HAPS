import importlib

import numpy as np
import openmdao.api as om
import longitudinal_kinematics as longitudinalkinematics
import potential_module
import aero_module
import battery_module



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


def test_potential_module():
    num_nodes = 5

    p = om.Problem(model=om.Group())

    ivc = p.model.add_subsystem('vars', om.IndepVarComp())
    ivc.add_output('h_dot', shape=(num_nodes,), units='m/s')

    p.model.add_subsystem('ode', potential_module.PotentialPower(num_nodes=num_nodes))

    p.model.connect('vars.h_dot', 'ode.h_dot')

    p.setup(force_alloc_complex=True)

    p.set_val('vars.h_dot', np.random.uniform(-5, 5, num_nodes))

    p.run_model()
    return p.check_partials(method='cs', compact_print=True)


def test_potential_module_totals():
    # Wires longitudinal_kinematics.Kinematics (h_dot = V*sin(gg)) straight into
    # potential_module.PotentialPower (Epot_sw = M_sw*g*h_dot) and checks the
    # TOTAL derivatives of Epot_sw wrt V and gg - the ones the optimizer actually
    # sees. This confirms OpenMDAO chains dEpot_sw/dh_dot * dh_dot/d{V,gg} through
    # the connection correctly, without potential_module re-deriving the V*sin(gg)
    # relation itself.
    num_nodes = 5

    p = om.Problem(model=om.Group())

    ivc = p.model.add_subsystem('vars', om.IndepVarComp())
    ivc.add_output('V', shape=(num_nodes,), units='m/s')
    ivc.add_output('gg', shape=(num_nodes,), units='rad')

    p.model.add_subsystem('kin', longitudinalkinematics.Kinematics(num_nodes=num_nodes))
    p.model.add_subsystem('pot', potential_module.PotentialPower(num_nodes=num_nodes))

    p.model.connect('vars.V', 'kin.V')
    p.model.connect('vars.gg', 'kin.gg')
    p.model.connect('kin.h_dot', 'pot.h_dot')

    p.setup(force_alloc_complex=True)

    p.set_val('vars.V', np.random.uniform(5, 15, num_nodes))
    p.set_val('vars.gg', np.radians(np.random.uniform(-10, 10, num_nodes)))

    p.run_model()
    return p.check_totals(of=['pot.Epot_sw'], wrt=['vars.V', 'vars.gg'], method='cs', compact_print=True)


def test_aero_module():
    num_nodes = 5

    p = om.Problem(model=om.Group())

    ivc = p.model.add_subsystem('vars', om.IndepVarComp())
    ivc.add_output('V', shape=(num_nodes,), units='m/s')
    ivc.add_output('h', shape=(num_nodes,), units='m')
    ivc.add_output('aa', shape=(num_nodes,), units='rad')
    # Tp is a raw throttle fraction (units=None on the component); gg is 'rad', same
    # as kinematics's gg, so the two can share a promoted name once wired into a Group.
    ivc.add_output('Tp', shape=(num_nodes,))
    ivc.add_output('gg', shape=(num_nodes,), units='rad')

    p.model.add_subsystem('ode', aero_module.DragPowerDissipation(num_nodes=num_nodes))

    for name in ['V', 'h', 'aa', 'Tp', 'gg']:
        p.model.connect(f'vars.{name}', f'ode.{name}')

    p.setup(force_alloc_complex=True)

    # V kept well away from 0 (Re = V*chord/kviscosity), h kept within the standard
    # atmosphere's range, same as test_3d_kinematics
    p.set_val('vars.V', np.random.uniform(5, 15, num_nodes))
    p.set_val('vars.h', np.random.uniform(15000, 24000, num_nodes))
    p.set_val('vars.aa', np.radians(np.random.uniform(0, 10, num_nodes)))
    p.set_val('vars.Tp', np.random.uniform(0, 1, num_nodes))
    p.set_val('vars.gg', np.radians(np.random.uniform(-10, 10, num_nodes)))

    p.run_model()
    # ambiance.Atmosphere (used in compute()) rejects complex input, so this can't
    # use complex-step - fd instead, same as aero_module's own declare_partials.
    return p.check_partials(method='fd', form='central', step=1e-6, compact_print=True)


def test_battery_module():
    num_nodes = 7

    p = om.Problem(model=om.Group())

    ivc = p.model.add_subsystem('vars', om.IndepVarComp())
    ivc.add_output('SOC', shape=(num_nodes,), units=None)
    ivc.add_output('Net_sw', shape=(num_nodes,), units='W/m**2')

    p.model.add_subsystem('ode', battery_module.StateOfCharge(num_nodes=num_nodes))

    p.model.connect('vars.SOC', 'ode.SOC')
    p.model.connect('vars.Net_sw', 'ode.Net_sw')

    p.setup(force_alloc_complex=True)

    # Deliberately straddles the SOC_max=0.8 saturation band (both sides, and net_sw of
    # both signs right at the cap) plus well away from it, since that's where the smooth
    # gate's partials are most at risk of a mistake.
    p.set_val('vars.SOC', [0.3, 0.6, 0.79, 0.8, 0.8, 0.81, 0.95])
    p.set_val('vars.Net_sw', [50.0, -30.0, 40.0, 50.0, -50.0, 20.0, -10.0])

    p.run_model()
    return p.check_partials(method='cs', compact_print=True)


if __name__ == '__main__':
    # test_lateral_kinematics()
    # test_3d_kinematics()
    # test_longitudinal_kinematics()
    # test_potential_module()
    # test_potential_module_totals()
    test_aero_module()
    # test_battery_module()
