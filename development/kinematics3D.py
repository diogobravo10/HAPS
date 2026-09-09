import numpy as np
import openmdao.api as om
from ambiance import Atmosphere

# Parameters
g = 9.81
m_sw = 3.0 # kg/m^2
chord = 1
T_installed = 100 # N/m^2


def aero_module(V, h, aa):


    # CD, CLmax = 0.0708, 1.2

    # CL and CD are fitted to a set of equations of the Reynolds number Re and the attack angle aa, 
    # where the Reynolds number is calculated according to current altitude and flight velocity
    rho = Atmosphere(h).density
    kviscosity = Atmosphere(h).kinematic_viscosity

    Re = V * chord / kviscosity

    # E216 low reynolds number airfoil
    a1, a2, a3, a4, a5, a6 = 3.77421e-1, 1.24316e-1, 7.64615e-7, -5.68228e-3, -6.44553e-13, -2.65058e-8
    b1, b2, b3, b4, b5, b6, b7, b8, b9 = 6.44815e-2, -1.87841e-7, 1.79326e-13, -1.11385e-2, 3.75046e-8, -3.10591e-14, 1.09753e-3, -2.36796e-9, 1.58461e-15

    CL = a1 + a2 * aa + a3 * Re + a4 * aa**2 + a5 * Re**2 + a6 * aa * Re
    CD = b1 + b2 * Re + b3 * Re**2 + b4 * aa + b5 * aa*Re + b6 * aa * Re**2 + b7 * aa**2 + b8 * aa**2 * Re + b9 * aa**2 * Re**2

    L_sw = 1/2 * rho * V**2 * CL
    D_sw =  1/2 * rho * V**2 * CD

    return D_sw, L_sw



class Kinematics3D(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs: States & Controls - used to compute outputs
        self.add_input('psi', val=np.zeros(nn), units='rad', desc='Heading angle')
        self.add_input('V', val=np.ones(nn), units='m/s', desc='Airspeed')
        self.add_input('gg', val=np.zeros(nn), units='rad', desc='Glide angle')
        self.add_input('h', val=np.zeros(nn), units='m', desc='Altitude')

        self.add_input('phi', val=np.zeros(nn), units='rad', desc='Bank angle (control)')
        self.add_input('aa', val=np.zeros(nn), units='rad', desc='Angle of attack (control)')
        self.add_input('Throttle', val=np.zeros(nn), units=None, desc='Throttle (control)')

        # Outputs: Variable to be integrated
        self.add_output('x_dot', val=np.zeros(nn), units='m/s')
        self.add_output('y_dot', val=np.zeros(nn), units='m/s')
        self.add_output('h_dot', val=np.zeros(nn), units='m/s')
        self.add_output('V_dot', val=np.zeros(nn), units='m/s**2')
        self.add_output('gg_dot', val=np.zeros(nn), units='rad/s')
        self.add_output('psi_dot', val=np.zeros(nn), units='rad/s')


        # Used to compute the derivatives of the outputs w.r.t. each of the inputs analytically or fd or cs
        arange = np.arange(self.options['num_nodes'])

        self.declare_partials(of='x_dot', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='x_dot', wrt='psi', rows=arange, cols=arange)
        self.declare_partials(of='x_dot', wrt='gg', rows=arange, cols=arange)

        self.declare_partials(of='y_dot', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='y_dot', wrt='psi', rows=arange, cols=arange)
        self.declare_partials(of='y_dot', wrt='gg', rows=arange, cols=arange)

        self.declare_partials(of='h_dot', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='h_dot', wrt='gg', rows=arange, cols=arange)
        # self.declare_partials(of='h_dot', wrt='psi', rows=arange, cols=arange)  # h_dot = V * sin(gg) has no psi dependence, so d(h_dot)/d(psi) is left undeclared (= 0)

        self.declare_partials(of='V_dot', wrt='*', method='cs')
        self.declare_partials(of='gg_dot', wrt='*', method='cs')
        self.declare_partials(of='psi_dot', wrt='*', method='cs')

        # ambiance.Atmosphere rejects complex input, so any partial touching 'h'
        # (V_dot, gg_dot, psi_dot all call aero_module -> Atmosphere(h)) needs fd instead
        self.declare_partials(of='V_dot', wrt='h', method='fd')
        self.declare_partials(of='gg_dot', wrt='h', method='fd')
        self.declare_partials(of='psi_dot', wrt='h', method='fd')






    def compute(self, inputs, outputs):
        # Used to compute the outputs, given the inputs.
        psi = inputs['psi']
        phi = inputs['phi']
        V = inputs['V']
        gg = inputs['gg']
        Throttle = inputs['Throttle'] * T_installed
        aa = inputs['aa']
        h = inputs['h']

        D_sw, L_sw = aero_module(V, h, aa)

        sin_psi, cos_psi = np.sin(psi), np.cos(psi)
        sin_gg, cos_gg = np.sin(gg), np.cos(gg)
        sin_phi, cos_phi = np.sin(phi), np.cos(phi)
        sin_aa, cos_aa = np.sin(aa), np.cos(aa)

        # Kinematic Dynamics
        outputs['x_dot'] = V * cos_psi * cos_gg
        outputs['y_dot'] = V * sin_psi * cos_gg
        outputs['h_dot'] = V * sin_gg
        outputs['V_dot'] = (Throttle * cos_aa - D_sw) / m_sw - g * sin_gg
        outputs['gg_dot'] = (Throttle * sin_aa + L_sw) * cos_phi / (V * m_sw) - g * cos_gg / V
        outputs['psi_dot'] = (Throttle * sin_aa + L_sw) * sin_phi / (V * m_sw * cos_gg)


    def compute_partials(self, inputs, partials):

        # Used to compute the derivatives of the outputs w.r.t. each of the inputs analytically
        psi = inputs['psi']
        V = inputs['V']
        gg = inputs['gg']

        sin_psi, cos_psi = np.sin(psi), np.cos(psi)
        sin_gg, cos_gg = np.sin(gg), np.cos(gg)

        partials['x_dot', 'V'] = cos_gg * cos_psi
        partials['x_dot', 'psi'] = -V * sin_psi * cos_gg
        partials['x_dot', 'gg'] = -V * cos_psi * sin_gg

        partials['y_dot', 'V'] = sin_psi * cos_gg
        partials['y_dot', 'psi'] = V * cos_psi * cos_gg
        partials['y_dot', 'gg'] = -V * sin_psi * sin_gg

        partials['h_dot', 'V'] = sin_gg
        partials['h_dot', 'gg'] = V * cos_gg
