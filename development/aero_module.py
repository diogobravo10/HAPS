import numpy as np
import openmdao.api as om
from ambiance import Atmosphere

# Parameters
g = 9.81
chord = 1 # dimensionalize by a 1meter
M_sw = 3.0 # [kg/m^2]
CD, CLmax = 0.0708, 1.2
Tinst_sw = 10 # N/m^2
mu_prop = 0.7 # Propeller efficiency depends on advance ratio ()

# E216 low reynolds number airfoil
a1, a2, a3, a4, a5, a6 = 3.77421e-1, 1.24316e-1, 7.64615e-7, -5.68228e-3, -6.44553e-13, -2.65058e-8
b1, b2, b3, b4, b5, b6, b7, b8, b9 = 6.44815e-2, -1.87841e-7, 1.79326e-13, -1.11385e-2, 3.75046e-8, -3.10591e-14, 1.09753e-3, -2.36796e-9, 1.58461e-15


class DragPowerDissipation(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs: States & Controls - used to compute outputs
        self.add_input('V', val=np.ones(nn), units='m/s', desc='Airspeed (control)')
        self.add_input('h', val=np.zeros(nn), units='m', desc='Altitude (state)')
        self.add_input('aa', val=np.zeros(nn), units='rad', desc='Angle of Attack (control)')
        self.add_input('Tp', val=np.zeros(nn), units=None, desc='Throttle (control)')
        self.add_input('gg', val=np.zeros(nn), units='rad', desc='Flight-path angle (control)')

        # Outputs: Variable to be integrated
        self.add_output('V_dot', val=np.zeros(nn), units='m/s**2')
        self.add_output('gg_dot', val=np.zeros(nn), units='rad/s')
        self.add_output('DV_sw', val=np.zeros(nn), units='W/m**2')
        self.add_output('TV_sw', val=np.zeros(nn), units='W/m**2')
        self.add_output('Vmargin', val=np.zeros(nn), units='m/s')

        # Used to compute the derivatives of the outputs w.r.t. each of the inputs analytically or fd or cs
        arange = np.arange(self.options['num_nodes'])

        # ambiance.Atmosphere (used in compute() below) rejects complex input, so these
        # can't use complex-step - fd instead, same as aero_module's other consumer
        # (kinematics3D.py's Kinematics3D).
        self.declare_partials(of='DV_sw', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='DV_sw', wrt='h', rows=arange, cols=arange, method='fd')
        self.declare_partials(of='DV_sw', wrt='aa', rows=arange, cols=arange)

        self.declare_partials(of='TV_sw', wrt='Tp', rows=arange, cols=arange)
        self.declare_partials(of='TV_sw', wrt='V', rows=arange, cols=arange)

        # Vstall depends on h only (wing loading/CLmax are fixed); Vmargin = V - Vstall.
        self.declare_partials(of='Vmargin', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='Vmargin', wrt='h', rows=arange, cols=arange, method='fd')

        self.declare_partials(of='V_dot', wrt='Tp', rows=arange, cols=arange)
        self.declare_partials(of='V_dot', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='V_dot', wrt='h', rows=arange, cols=arange, method='fd')
        self.declare_partials(of='V_dot', wrt='aa', rows=arange, cols=arange)
        self.declare_partials(of='V_dot', wrt='gg', rows=arange, cols=arange)

        self.declare_partials(of='gg_dot', wrt='V', rows=arange, cols=arange)
        self.declare_partials(of='gg_dot', wrt='h', rows=arange, cols=arange, method='fd')
        self.declare_partials(of='gg_dot', wrt='aa', rows=arange, cols=arange)
        self.declare_partials(of='gg_dot', wrt='gg', rows=arange, cols=arange)


    def compute(self, inputs, outputs):
        # Used to compute the outputs, given the inputs.
        V = inputs['V']
        h = inputs['h']
        aa = np.degrees(inputs['aa'])
        Tp = inputs['Tp']
        gg = inputs['gg']

        sin_gg = np.sin(gg)
        cos_gg = np.cos(gg)
        cos_phi = 1

        # CL and CD are fitted to a set of equations of the Reynolds number Re and the attack angle aa, 
        # where the Reynolds number is calculated according to current altitude and flight velocity
        atm = Atmosphere(h)
        rho = atm.density
        kviscosity = atm.kinematic_viscosity

        Re = V * chord / kviscosity

        CL = a1 + a2 * aa + a3 * Re + a4 * aa**2 + a5 * Re**2 + a6 * aa * Re
        CD = b1 + b2 * Re + b3 * Re**2 + b4 * aa + b5 * aa*Re + b6 * aa * Re**2 + b7 * aa**2 + b8 * aa**2 * Re + b9 * aa**2 * Re**2

        L_sw = 1/2 * rho * V**2 * CL
        D_sw =  1/2 * rho * V**2 * CD

        # Stall speed
        Vstall = np.sqrt(2 * M_sw * g / rho / CLmax)

        outputs['DV_sw'] = D_sw * V 
        outputs['TV_sw'] = Tp * Tinst_sw * V /mu_prop
        outputs['Vmargin'] = V - 1.2*Vstall
        outputs['V_dot'] = (Tp*Tinst_sw - D_sw) / M_sw - g*sin_gg
        outputs['gg_dot'] = L_sw / (V* M_sw) * cos_phi - g*cos_gg/V


    def compute_partials(self, inputs, partials):


        nn = self.options['num_nodes']
       
        # Used to compute the derivatives of the outputs w.r.t. each of the inputs analytically
        V = inputs['V']
        h = inputs['h']
        aa = np.degrees(inputs['aa'])
        Tp = inputs['Tp']
        gg = inputs['gg']

        cos_gg = np.cos(gg)
        sin_gg = np.sin(gg)
        cos_phi = 1

        atm = Atmosphere(h)
        rho = atm.density
        kviscosity = atm.kinematic_viscosity

        Re = V * chord / kviscosity

        partials['DV_sw', 'V'] = 1/2 * rho * V**2 * (3*(b1 + b4*aa + b7*aa**2) + 4*(b2 + b5*aa + b8*aa**2)*Re + 5*(b3 + b6*aa + b9*aa**2)*Re**2)
        # - chain rule: d(aa_deg)/d(aa_rad) = 180/pi.
        partials['DV_sw', 'aa'] = 1/2 * rho * V**3 * (b4 + b5*Re + b6*Re**2 + 2*b7*aa + 2*b8*aa*Re + 2*b9*aa*Re**2) * (180/np.pi)

        partials['TV_sw', 'Tp'] =  Tinst_sw * V /mu_prop
        partials['TV_sw', 'V'] = Tp * Tinst_sw /mu_prop

        partials['Vmargin', 'V'] = np.ones(nn)

        partials['V_dot', 'Tp'] = Tinst_sw/M_sw
        partials['V_dot', 'gg'] = -g* cos_gg        
        partials['V_dot', 'V'] = -rho * V / (2*M_sw) * (2*(b1 + b4*aa + b7*aa**2) + 3*(b2 + b5*aa + b8*aa**2)*Re + 4*(b3 + b6*aa + b9*aa**2)*Re**2)
        partials['V_dot', 'aa'] = -rho * V**2 / (2*M_sw) * (b4 + b5*Re + b6*Re**2 + 2*b7*aa + 2*b8*aa*Re + 2*b9*aa*Re**2) * (180/np.pi)

        partials['gg_dot', 'gg'] = g* sin_gg / V       
        partials['gg_dot', 'aa'] = rho * V * cos_phi/ (2*M_sw) * (a2 + 2*a4*aa + a6*Re) * (180/np.pi)
        partials['gg_dot', 'V'] = g*cos_gg/V**2 + rho* cos_phi / (2*M_sw) * (a1 + a2*aa + 2*a3*Re + a4*aa**2 + 3*a5*Re**2 + 2*a6*aa*Re)