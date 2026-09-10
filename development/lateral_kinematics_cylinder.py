import numpy as np
import openmdao.api as om

# Parameters
g = 9.81


class Kinematics(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs: States & Controls - used to compute outputs
        self.add_input('tt', val=np.zeros(nn), units='rad', desc='Angular position along cylinder (state)')
        self.add_input('V', val=np.ones(nn), units='m/s', desc='Airspeed (control)')
        self.add_input('R', val=np.zeros(nn), units='m', desc='Turn radius (parameter)')

        # Outputs: Variable to be integrated
        self.add_output('tt_dot', val=np.zeros(nn), units='rad/s')
        self.add_output('x', val=np.zeros(nn), units='m')
        self.add_output('y', val=np.zeros(nn), units='m')

        # Used to compute the derivatives of the outputs w.r.t. each of the inputs analytically or fd or cs
        arange = np.arange(self.options['num_nodes'])

        self.declare_partials(of='tt_dot', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='tt_dot', wrt='R', rows=arange, cols=arange)

        self.declare_partials(of='x', wrt='tt', rows=arange, cols=arange)
        self.declare_partials(of='x', wrt='R', rows=arange, cols=arange)

        self.declare_partials(of='y', wrt='tt', rows=arange, cols=arange)
        self.declare_partials(of='y', wrt='R', rows=arange, cols=arange)


    def compute(self, inputs, outputs):
        # Used to compute the outputs, given the inputs.
        V = inputs['V']
        R = inputs['R']
        tt = inputs['tt']

        sin_tt, cos_tt = np.sin(tt), np.cos(tt)

        # Kinematic Dynamics
        outputs['tt_dot'] = V / R
        outputs['x'] = R*cos_tt
        outputs['y'] = R*sin_tt
        # outputs['psi'] = tt + np.pi/2
        # outputs['phi'] = np.arctan(V**2/(g * R))

    def compute_partials(self, inputs, partials):

        # Used to compute the derivatives of the outputs w.r.t. each of the inputs analytically
        V = inputs['V']
        R = inputs['R']
        tt = inputs["tt"]

        sin_tt, cos_tt = np.sin(tt), np.cos(tt)

        partials['tt_dot', 'V'] = 1 / R
        partials['tt_dot', 'R'] = -V / (R**2)

        partials['x', 'tt'] = -R*sin_tt
        partials['x', 'R'] = cos_tt
        
        partials['y', 'tt'] = R*cos_tt
        partials['y', 'R'] = sin_tt
