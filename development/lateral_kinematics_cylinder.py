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
        self.add_input('psi', val=np.zeros(nn), units='rad', desc='Heading angle')
        self.add_input('phi', val=np.zeros(nn), units='rad', desc='Bank angle (control)')
        self.add_input('V', val=np.ones(nn), units='m/s', desc='Airspeed (control)')

        # Outputs: Variable to be integrated
        self.add_output('x_dot', val=np.zeros(nn), units='m/s')
        self.add_output('y_dot', val=np.zeros(nn), units='m/s')
        self.add_output('psi_dot', val=np.zeros(nn), units='rad/s')

        # Used to compute the derivatives of the outputs w.r.t. each of the inputs analytically or fd or cs
        arange = np.arange(self.options['num_nodes'])

        self.declare_partials(of='x_dot', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='x_dot', wrt='psi', rows=arange, cols=arange)

        self.declare_partials(of='y_dot', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='y_dot', wrt='psi', rows=arange, cols=arange)

        self.declare_partials(of='psi_dot', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='psi_dot', wrt='phi', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        # Used to compute the outputs, given the inputs.
        psi = inputs['psi']
        phi = inputs['phi']
        V = inputs['V']

        # Kinematic Dynamics
        outputs['x_dot'] = V * np.cos(psi)
        outputs['y_dot'] = V * np.sin(psi)
        outputs['psi_dot'] = (g * np.tan(phi)) / V

    def compute_partials(self, inputs, partials):

        # Used to compute the derivatives of the outputs w.r.t. each of the inputs analytically
        psi = inputs['psi']
        phi = inputs['phi']
        V = inputs['V']

        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        partials['x_dot', 'V'] = cos_psi
        partials['x_dot', 'psi'] = -V * sin_psi

        partials['y_dot', 'V'] = sin_psi
        partials['y_dot', 'psi'] = V * cos_psi

        partials['psi_dot', 'V'] = -(g * np.tan(phi)) / (V**2)
        partials['psi_dot', 'phi'] = (g / V) / (np.cos(phi)**2)
