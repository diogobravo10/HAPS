import numpy as np
import openmdao.api as om


class StateOfCharge(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('mbat_sw', default=2.0, types=(int, float), desc='Battery mass per unit wing area [kg/m^2]')
        self.options.declare('mb', default=450.0, types=(int, float), desc='Battery energy density [Wh/kg]')
        self.options.declare('mu_e', default=0.9, types=(int, float), desc='Energy management system efficiency')
        self.options.declare('mu_LS', default=0.9, types=(int, float), desc='LS-battery efficiency')

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs: States & Controls - used to compute outputs
        self.add_input('SOC', val=np.zeros(nn), units=None, desc='State of Charge (stste)')
        self.add_input('Net_sw', val=np.zeros(nn), units='W/m**2', desc='Net Energy (stste)')

        # Outputs: Variable to be integrated - rate of change of the state of charge
        # (dimensionless fraction, 0-1), same naming convention as h_dot/V_dot.
        self.add_output('SOC_dot', val=np.zeros(nn), units='1/s')

        # Used to compute the derivatives of the outputs w.r.t. each of the inputs analytically or fd or cs
        arange = np.arange(self.options['num_nodes'])

        self.declare_partials(of='SOC_dot', wrt='Net_sw', rows=arange, cols=arange)



    def compute(self, inputs, outputs):
        # Used to compute the outputs, given the inputs.
        mbat_sw = self.options['mbat_sw']
        mb = self.options['mb']

        Net_sw = inputs['Net_sw']

        # Battery energy capacity per unit wing area (mb is Wh/kg -> J/m**2 via *3600).
        max_energy = mb * 60*60 * mbat_sw

        outputs['SOC_dot'] = Net_sw / max_energy

        # outputs['SOC_dot'] = np.where((SOC > SOC_max) & (Net_sw > 0), 0.0, Net_sw / max_energy)


    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        mbat_sw = self.options['mbat_sw']
        mb = self.options['mb']
        max_energy = mb * 60*60 * mbat_sw

        partials['SOC_dot', 'Net_sw'] = np.ones(nn) / max_energy


