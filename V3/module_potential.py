import numpy as np
import openmdao.api as om


class PotentialPower(om.ExplicitComponent):
    def initialize(self):
        # Dymos requires num_nodes to vectorize calculations across the trajectory grid
        self.options.declare('num_nodes', types=int)
        self.options.declare('g', default=9.81, types=(int, float), desc='Gravitational acceleration [m/s^2]')
        self.options.declare('M_sw', default=3.0, types=(int, float), desc='Wing loading (mass per unit wing area) [kg/m^2]')

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs: States & Controls
        self.add_input('h_dot', val=np.ones(nn), units='m/s', desc='Rate of change of altitude')

        # Outputs: Variable to be integrated - M_sw is per unit wing area (kg/m^2), so
        # M_sw*g*h_dot comes out in W/m**2, same convention as aero_module's DV_sw and
        # solar_module's Psol_sw (not plain W).
        self.add_output('Epot_sw', val=np.zeros(nn), units='W/m**2',
                         desc='Potential-energy accumulation rate per unit wing area')

        arange = np.arange(nn)
        self.declare_partials(of='Epot_sw', wrt='h_dot', rows=arange, cols=arange)


    def compute(self, inputs, outputs):
        g = self.options['g']
        M_sw = self.options['M_sw']

        h_dot = inputs['h_dot']

        # Potential power per unit wing area
        outputs['Epot_sw'] = M_sw*g*h_dot

    def compute_partials(self, inputs, partials):

        nn = self.options['num_nodes']
        g = self.options['g']
        M_sw = self.options['M_sw']

        partials['Epot_sw', 'h_dot'] = M_sw*g*np.ones(nn)
