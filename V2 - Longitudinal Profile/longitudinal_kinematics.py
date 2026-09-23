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
        self.add_input('V', val=np.ones(nn), units='m/s', desc='Airspeed (control)')
        self.add_input('gg', val=np.zeros(nn), units='rad', desc='Flight-path angle (control)')
        
        # Outputs: Variable to be integrated
        self.add_output('h_dot', val=np.zeros(nn), units='m/s')


        # Used to compute the derivatives of the outputs w.r.t. each of the inputs analytically or fd or cs
        arange = np.arange(self.options['num_nodes'])

        self.declare_partials(of='h_dot', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='h_dot', wrt='gg', rows=arange, cols=arange)


    def compute(self, inputs, outputs):
        # Used to compute the outputs, given the inputs.
        gg = inputs['gg']
        V = inputs['V']

        sin_gg = np.sin(gg)

        # Kinematic Dynamics
        outputs['h_dot'] = V * sin_gg


    def compute_partials(self, inputs, partials):

        # Used to compute the derivatives of the outputs w.r.t. each of the inputs analytically
        gg = inputs['gg']
        V = inputs['V']

        cos_gg = np.cos(gg)
        sin_gg = np.sin(gg)

        partials['h_dot', 'V'] = sin_gg
        partials['h_dot', 'gg'] = V * cos_gg

