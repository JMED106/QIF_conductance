import numpy as np
import logging

try:
    import numba
except:
    raise ImportError('Install numba. Available in pip, anaconda, etc.\nTested version 0.13.0')

if numba.__version__ != '0.13.0':
    logging.warning('Numba verision %s not tested, it may have unexpected behavior.' % numba.__version__)
    pass


__author__ = 'Jose M. Esnaola Acebes'

""" Library containing different models of spiking neurons:

    + QIF, QIF + noise. INTEGRATION

"""


# -- Spiking neurons with euler integration.
# ------------------------------------------

# Deterministic QIF neuron for numba
# @numba.autojit4
@numba.jit()
def qifint(v_exit_s1, v, exit0, eta_0, s_0, tiempo, number, dn, dt, tau, vpeak, refr_tau, tau_peak):
    """ This function checks (for each neuron) whether the neuron is in the
    refractory period, and computes the integration in case is NOT. If it is,
    then it adds a time step until the refractory period finishes.

    The spike is computed when the neuron in the refractory period, i.e.
    a neuron that has already crossed the threshold, reaches the midpoint
    in the refractory period, t_peak.
    :rtype : object
    """

    d = 1 * v_exit_s1
    # These steps are necessary in order to use Numba
    t = tiempo * 1.0
    for n in xrange(number):
        d[n, 2] = 0
        if t >= exit0[n]:
            d[n, 0] = v[n] + (dt / tau) * (v[n] * v[n] + eta_0[n] + tau * s_0[int(n / dn)])  # Euler integration
            if d[n, 0] >= vpeak:
                d[n, 1] = t + refr_tau - (tau_peak - 1.0 / d[n, 0])
                d[n, 2] = 1
                d[n, 0] = -d[n, 0]
    return d


# Deterministic QIF neuron with conductance based dynamics for numba
# @numba.autojit4
@numba.jit()
def qifint_cond(v_exit_s1, v, exit0, eta_0, s_0, tiempo, number, dt, tau, vpeak, refr_tau, tau_peak, reversal, g):
    """ This function checks (for each neuron) whether the neuron is in the
    refractory period, and computes the integration in case is NOT. If it is,
    then it adds a time step until the refractory period finishes.

    The spike is compute
    d when the neuron in the refractory period, i.e.
    a neuron that has already crossed the threshold, reaches the
     midpoint
    in the refractory period, t_peak.
    :rtype : object
    """

    d = 1 * v_exit_s1
    # These steps are necessary in order to use Numba
    t = tiempo * 1.0
    for n in xrange(number):
        d[n, 2] = 0
        if t >= exit0[n]:
            # Euler integration
            d[n, 0] = v[n] + (dt / tau) * (v[n] * v[n] + eta_0[n] - tau * g * s_0 * (v[n] - reversal))
            if d[n, 0] >= vpeak:
                d[n, 1] = t + refr_tau - (tau_peak - 1.0 / d[n, 0])
                d[n, 2] = 1
                d[n, 0] = -d[n, 0]
    return d


# Noisy QIF neuron for numba
@numba.jit()
def qifint_noise(v_exit_s1, v, exit0, eta_0, s_0, nois, tiempo, number, dn, dt, tau, vpeak, refr_tau, tau_peak):
    d = 1 * v_exit_s1
    # These steps are necessary in order to use Numba (don't ask why ...)
    t = tiempo * 1.0
    for n in xrange(number):
        d[n, 2] = 0
        if t >= exit0[n]:
            d[n, 0] = v[n] + (dt / tau) * (v[n] * v[n] + eta_0 + tau * s_0[int(n / dn)]) + nois[n]  # Euler integration
            if d[n, 0] >= vpeak:
                d[n, 1] = t + refr_tau - (tau_peak - 1.0 / d[n, 0])
                d[n, 2] = 1
                d[n, 0] = -d[n, 0]
    return d


# - - Distributions
# Lorentz distribution
def lorentz(n, center, width):
    k = (2.0 * np.arange(1, n + 1) - n - 1.0) / (n + 1.0)
    y = center + width * np.tan((np.pi / 2.0) * k)
    return y
