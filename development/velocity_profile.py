import numpy as np
import openmdao.api as om
from scipy.interpolate import CubicSpline


class VelocityProfileComp(om.ExplicitComponent):
    """Evaluates a precomputed V(t) profile (e.g. from a solved longitudinal
    trajectory) at whatever time value the calling phase currently needs,
    via a cubic spline. V is a derived output here, not a design variable -
    the analytic partial wrt time is the spline's own derivative."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('t_data', desc='1D array of time samples')
        self.options.declare('V_data', desc='1D array of V samples at t_data')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('time', val=np.zeros(nn), units='s')
        self.add_output('V', val=np.ones(nn), units='m/s')

        # Stitched multi-phase timeseries data can have an exact duplicate time
        # value at each phase boundary; CubicSpline requires strictly increasing x.
        t_data = np.asarray(self.options['t_data'])
        V_data = np.asarray(self.options['V_data'])
        keep = np.concatenate([[True], np.diff(t_data) > 0])
        t_data, V_data = t_data[keep], V_data[keep]

        self._spline = CubicSpline(t_data, V_data)
        self._dspline = self._spline.derivative()

        arange = np.arange(nn)
        self.declare_partials(of='V', wrt='time', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        outputs['V'] = self._spline(inputs['time'])

    def compute_partials(self, inputs, partials):
        partials['V', 'time'] = self._dspline(inputs['time'])
