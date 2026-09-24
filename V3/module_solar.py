import os
import sys

import numpy as np
import openmdao.api as om
from ambiance import Atmosphere
from datetime import datetime

# _utilities.py lives at the repo root, one level up from this development/ folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _utilities as utils

class SolarPower(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        # Reference date/time and latitude for the solar model (t=0 of the phase)
        self.options.declare('start_date', types=datetime)
        self.options.declare('lat', types=(int, float))
        self.options.declare('solar_cell_efficiency', default=0.15, types=(int, float))
        # -> update based on bank angle
        self.options.declare('vnorm', default=np.array([0, 0, -1]))

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs: States & Controls - used to compute outputs
        self.add_input('h', val=np.zeros(nn), units='m', desc='Altitude (state)')
        self.add_input('time', val=np.zeros(nn), units='s')

        # Outputs: Variable to be integrated
        self.add_output('Psol_sw', val=np.zeros(nn), units='W/m**2')

        # Used to compute the derivatives of the outputs w.r.t. each of the inputs analytically or fd or cs
        arange = np.arange(self.options['num_nodes'])

        self.declare_partials(of='Psol_sw', wrt='h', rows=arange, cols=arange, method='fd')
        # The solar model is quantized to whole minutes internally (solarpy's hour_angle
        # only reads date.hour/date.minute), so the default FD step on 'time' (~1e-6 x a
        # few thousand seconds) is far smaller than 60s and would yield a numerically zero
        # d(Psol_sw)/d(time) - same issue as SolarAircraftODE in the root longitudinal_kinematics.py.
        self.declare_partials(of='Psol_sw', wrt='time', rows=arange, cols=arange, method='fd', step=60.0, step_calc='abs')


    def compute(self, inputs, outputs):
        # Used to compute the outputs, given the inputs.
        h = inputs['h']
        t = inputs['time']

        Psol_sw = utils.instantaneous_power_density_vect(
            h, self.options['lat'], self.options['start_date'], t,
            solar_cell_efficiency=self.options['solar_cell_efficiency'], vnorm=self.options['vnorm'])
        outputs['Psol_sw'] = Psol_sw


