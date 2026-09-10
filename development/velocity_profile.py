import numpy as np
import openmdao.api as om
from scipy.interpolate import PchipInterpolator


class VelocityProfileComp(om.ExplicitComponent):
    """Evaluates a precomputed V(t) profile (e.g. from a solved longitudinal
    trajectory) at whatever time value the calling phase currently needs,
    via a shape-preserving spline. V is a derived output here, not a design
    variable - the analytic partial wrt time is the spline's own derivative."""

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('t_data', desc='1D array of time samples')
        self.options.declare('V_data', desc='1D array of V samples at t_data')
        self.options.declare('V_min', default=1.0,
                              desc='Floor applied to the interpolated V, so that '
                                   'g*tan(phi)/V cannot blow up if the spline dips '
                                   'near zero between samples')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('time', val=np.zeros(nn), units='s')
        self.add_output('V', val=np.ones(nn), units='m/s')

        # Stitched multi-phase timeseries data can have an exact duplicate time
        # value at each phase boundary; the interpolant requires strictly increasing x.
        t_data = np.asarray(self.options['t_data'])
        V_data = np.asarray(self.options['V_data'])
        keep = np.concatenate([[True], np.diff(t_data) > 0])
        t_data, V_data = t_data[keep], V_data[keep]

        # PchipInterpolator is shape-preserving (no overshoot/undershoot between
        # samples), unlike CubicSpline, which can ring below the true data range
        # near a kink (e.g. a phase boundary) and drive V toward zero or negative.
        self._spline = PchipInterpolator(t_data, V_data)
        self._dspline = self._spline.derivative()

        arange = np.arange(nn)
        self.declare_partials(of='V', wrt='time', rows=arange, cols=arange)

    def compute(self, inputs, outputs):
        V = self._spline(inputs['time'])
        outputs['V'] = np.maximum(V, self.options['V_min'])

    def compute_partials(self, inputs, partials):
        time = inputs['time']
        V = self._spline(time)
        dV = self._dspline(time)
        # Where the floor is active, V is constant w.r.t. time -> zero slope.
        partials['V', 'time'] = np.where(V > self.options['V_min'], dV, 0.0)
