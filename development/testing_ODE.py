import numpy as np
import openmdao.api as om
import lateral_kinematics as kinematics


num_nodes = 5

p = om.Problem(model=om.Group())

ivc = p.model.add_subsystem('vars', om.IndepVarComp())
ivc.add_output('V', shape=(num_nodes,), units='m/s')
ivc.add_output('phi', shape=(num_nodes,), units='deg')

p.model.add_subsystem('ode', kinematics.Kinematics(num_nodes=num_nodes))

p.model.connect('vars.V', 'ode.V')
p.model.connect('vars.phi', 'ode.phi')

p.setup(force_alloc_complex=True)

p.set_val('vars.V', 10*np.random.random(num_nodes))
p.set_val('vars.phi', 10*np.random.uniform(1, 179, num_nodes))

p.run_model()
cpd = p.check_partials(method='cs', compact_print=True)