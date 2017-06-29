import numpy as np
import logging
import psutil
import spneuron

logging.getLogger('simu_lib').addHandler(logging.NullHandler())

__author__ = 'Jose M. Esnaola Acebes'

""" This file contains classes and functions to be used in the QIF network simulation.

    Data: (to store parameters, variables, and some functions)
    *****
"""


class Data:
    def __init__(self, n=1E5, eta0=0, g=0.0, revpot=0.0, delta=1.0, t0=0.0, tfinal=50.0,
                 dt=1E-3, tau=1.0, faketau=20.0E-3, fp='lorentz', system='qif'):
        self.logger = logging.getLogger('nflib.Data')
        self.logger.debug("Creating data structure.")
        # Zeroth mode, determines firing rate of the homogeneous state
        self.g = g  # default value
        self.reverv = revpot

        # 0.3) Give the model parameters
        self.eta0 = eta0  # Constant external current mean value
        self.delta = delta  # Constant external current distribution width

        # 0.2) Define the temporal resolution and other time-related variables
        self.t0 = t0  # Initial time
        self.tfinal = tfinal  # Final time
        self.total_time = tfinal - t0  # Time of simulation
        self.dt = dt  # Time step

        self.tpoints = np.arange(t0, tfinal, dt)  # Points for the plots and others
        self.nsteps = len(self.tpoints)  # Total time steps
        self.tau = tau
        self.faketau = faketau  # time scale in ms

        self.sys = system
        self.systems = []
        if system == 'qif' or system == 'both':
            self.systems.append('qif')
        if system == 'nf' or system == 'both':
            self.systems.append('nf')

        self.logger.debug("Simulating %s system(s)." % self.systems)

        # QIF model parameters
        if system != 'nf':
            self.logger.info("Loading QIF parameters:")
            self.fp = fp
            # sub-populations
            self.N = np.int(n)
            self.auxne = np.ones((1, self.N))
            # sub-populations

            self.vpeak = 100.0  # Value of peak voltage (max voltage)
            self.vreset = -100.0
            # --------------
            self.refr_tau = tau / self.vpeak - tau / self.vreset  # Refractory time in which the neuron is not excitable
            self.tau_peak = tau / self.vpeak  # Refractory time till the spike is generated
            # --------------
            self.T_syn = 10  # Number of steps for computing synaptic activation
            self.tau_syn = self.T_syn * dt  # time scale (??)

            # We need T_syn vectors in order to improve the performance.
            if self.T_syn == 10:
                # Heaviside
                h_tau = 1.0 / self.tau_syn
                a_tau0 = np.transpose(h_tau * np.ones(self.T_syn))
            else:
                self.logger.error("Not implemented... exiting.")
                self.h_tau = 0.0
                a_tau0 = 0.0
                exit(-1)

            self.a_tau = np.zeros((self.T_syn, self.T_syn))  # Multiple weighting vectors (index shifted)
            for i in xrange(self.T_syn):
                self.a_tau[i] = np.roll(a_tau0, i, 0)

            # Distributions of the external current       -- FOR l populations --
            self.eta = None
            if fp == 'lorentz' or fp == 'gauss':
                self.logger.info("+ Setting distribution of external currents: ")
                if fp == 'lorentz':
                    self.logger.info('   - Lorentzian distribution of external currents')
                    self.eta = spneuron.lorentz(self.N, self.eta0, self.delta)
                else:
                    self.logger.info('   - Gaussian distribution of external currents')
                    self.eta = None
            elif fp == 'noise':
                self.logger.info("+ Setting homogeneous population of neurons (identical), under GWN.")
                self.eta = np.ones(self.N) * self.eta0
            else:
                self.logger.critical("This distribution is not implemented, yet.")
                exit(-1)

            # QIF neurons matrices (declaration)
            self.spiketime = int(self.tau_peak / dt)
            self.s1time = self.T_syn + self.spiketime
            self.matrix, self.spikes, self.spikes_mod = self.load()
            if not np.any(self.matrix):
                self.logger.debug("Creating new initial conditions ...")
                self.matrix = np.ones(shape=(self.N, 3)) * 0
                self.spikes = np.ones(shape=(self.N, self.T_syn)) * 0  # Spike matrix (N x T_syn)
                self.spikes_mod = np.ones(shape=(self.N, self.spiketime)) * 0  # Spike matrix (N x (T_syn + tpeak/dt))

                # 0.8.1) QIF vectors and matrices (initial conditions are loaded after
                #                                  connectivity  matrix is created)

        if system != 'qif':
            self.r = np.ones(self.nsteps) * 0.1
            self.v = np.ones(self.nsteps) * (-0.01)
            self.r[len(self.r) - 1] = 0.1

    def save(self, rolling):
        np.save("neurons.npy", self.matrix)
        # np.roll(self.spikes, rolling)
        np.save("spikes.npy", self.spikes)
        np.save("spikes_mod.npy", self.spikes_mod)

    def load(self):
        try:
            matrix = np.load("neurons.npy")
            spikes = np.load("spikes.npy")
            spikes_mod = np.load("spikes_mod.npy")
            self.logger.debug("Successfully loaded initial conditions.")
            return matrix, spikes, spikes_mod
        except:
            return None, None, None


class FiringRate:
    """ Class related to the measure of the firing rate of a neural network.
    """

    def __init__(self, data=None, swindow=1.0, sampling=0.01, points=None):

        self.log = logging.getLogger('nflib.FiringRate')
        if data is None:
            self.d = Data()
        else:
            self.d = data

        self.swindow = swindow  # Time Window in which we measure the firing rate
        self.wsteps = int(np.ceil(self.swindow / self.d.dt))  # Time window in time-steps
        self.wones = np.ones(self.wsteps)

        # Frequency of measuremnts
        if points is not None:
            pp = points
            if pp > self.d.nsteps:
                pp = self.d.nsteps
            self.sampling = self.d.nsteps / pp
            self.samplingtime = self.sampling * self.d.dt
        else:
            self.samplingtime = sampling
            self.sampling = int(self.samplingtime / self.d.dt)

        # Firing rate of single neurons (distibution of firing rates)
        self.sampqift = 1.0 * self.swindow
        self.sampqif = int(self.sampqift / self.d.dt)

        self.tpoints_r = np.arange(0, self.d.tfinal, self.samplingtime)

        freemem = psutil.virtual_memory().available
        needmem = 8 * self.wsteps * data.N
        self.log.info("Approximately %d MB of memory will be allocated for FR measurement." % (needmem / (1024 ** 2)))
        if (freemem - needmem) / (1024 ** 2) <= 0:
            self.log.error("MEMORY ERROR: not enough amount of memory available.")
            exit(-1)
        elif (freemem - needmem) / (1024 ** 2) < 100:
            self.log.warning("CRITICAL WARNING: very few amount of memory will be left.")
            try:
                raw_input("Continue? (any key to continue, CTRL+D to terminate).")
            except EOFError:
                self.log.critical("Terminating process.")
                exit(-1)

        self.frspikes = 0 * np.zeros(shape=(data.N, self.wsteps))  # Secondary spikes matrix (for measuring)
        self.r = []  # Firing rate of the newtork(ring)
        self.rqif = []
        self.v = []  # Firing rate of the newtork(ring)
        self.vavg_e = 0.0 * np.ones(data.N)
        self.frqif = []  # Firing rate of individual qif neurons
        self.frqif = None

        # Total spikes of the network:
        self.tspikes = 0 * np.ones(data.N)
        self.tspikes2 = 0 * np.ones(data.N)

        # Theoretical distribution of firing rates
        self.thdist = dict()

        # Auxiliary counters
        self.ravg = 0
        self.ravg2 = 0
        self.vavg = 0

        # Times of firing rate measures
        self.t0step = None
        self.tfstep = None
        self.temps = None

        self.tfrstep = -1
        self.tfr = []
        self.tempsfr = []
        self.tempsfr2 = []

    def firingrate(self, tstep):
        """ Computes the firing rate for a given matrix of spikes. Firing rate is computed
            every certain time (sampling). Therefore at some time steps the firing rate is not computed,
        :param tstep: time step of the simulation
        :return: firing rate vector (matrix)
        """
        if (tstep + 1) % self.sampling == 0 and (tstep * self.d.dt >= self.swindow):
            self.tfrstep += 1
            self.temps = tstep * self.d.dt
            re = (1.0 / self.swindow) * (1.0 / self.d.N) * np.sum(
                np.dot(self.frspikes, self.wones))
            self.r.append(re)
            self.tempsfr.append(self.temps - self.swindow / 2.0)
            self.tempsfr2.append(self.temps)

            # Reset vectors and counter
            self.tspikes2 = 0.0 * np.ones(self.d.N)
            self.ravg2 = 0

    def singlefiringrate(self, tstep):
        """ Computes the firing rate of individual neurons.
        :return: Nothing, results are stored at frqif0 and frqif_i
        """
        if (tstep + 1) % self.sampqif == 0 and (tstep * self.d.dt >= self.swindow):
            # Firing rate measure in a time window
            re = (1.0 / self.d.dt) * self.frspikes.mean(axis=1)
            self.frqif.append(re)
