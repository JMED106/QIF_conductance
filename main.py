#!/usr/bin/python2.7

import numpy as np
from sconf import parser_init, parser, log_conf
from spneuron import qifint_cond
from simu_lib import Data, FiringRate
import progressbar as pb
from timeit import default_timer as timer

import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pylab as plt

import Gnuplot

# Use this option to turn off fifo if you get warnings like:
# line 0: warning: Skipping unreadable file "/tmp/tmpakexra.gnuplot/fifo"
Gnuplot.GnuplotOpts.prefer_fifo_data = 0

__author__ = 'jm'


# Empty class to manage external parameters
# noinspection PyClassHasNoInit
class Options:
    pass


options = None
ops = Options()
pi = np.pi
pi2 = np.pi * np.pi

# -- Simulation configuration I: parsing, debugging.
conf_file, debug, args1, hlp = parser_init()
if not hlp:
    logger = log_conf(debug)
else:
    logger = None
# -- Simulation configuration II: data entry (second parser).
description = 'Conductance based QIF spiking neural network. All to all coupled with distributed external currents.'
opts, args = parser(conf_file, args1, description=description)  # opts is a dictionary, args is an object
# Parameters are now those introduced in the configuration file:
# >>> args.parameter1 + args.parameter2
d = Data(args.N, args.e, args.g, args.E, args.d, args.t0, args.tf, args.dt, 1.0, args.tm, args.D, args.s)
fr = FiringRate(data=d, swindow=0.1, sampling=0.05)
# Progress-bar configuration
widgets = ['Progress: ', pb.Percentage(), ' ',
           pb.Bar(marker=pb.RotatingMarker()), ' ', pb.ETA(), ' ']

# ############################################################
# 0) Prepare simulation environment
pbar = pb.ProgressBar(widgets=widgets, maxval=10 * (d.nsteps + 1)).start()
time1 = timer()
tstep = 0
temps = 0
kp = k = 0
# Time loop
while temps < d.tfinal:
    # TIme step variables
    kp = tstep % d.nsteps
    k = (tstep + d.nsteps - 1) % d.nsteps
    k2p = tstep % 2
    k2 = (tstep + 2 - 1) % 2

    if tstep % 1000 == 0:
        d.eta0 -= 0.00
        d.g += 0.02
        logger.debug("Eta: %f\tg: %f" % (d.eta0, d.g))
    if d.sys in ('qif', 'both'):
        tsyp = tstep % d.T_syn
        tskp = tstep % d.spiketime
        tsk = (tstep + d.spiketime - 1) % d.spiketime
        # We compute the Mean-field vector s_j
        s = (1.0 / d.N) * np.sum(np.dot(d.spikes, d.a_tau[:, tsyp]))

        # Integration
        d.matrix = qifint_cond(d.matrix, d.matrix[:, 0], d.matrix[:, 1], d.eta, s, temps, d.N, d.dt, d.tau,
                               d.vpeak, d.refr_tau, d.tau_peak, d.reverv, d.g)

        # Prepare spike matrices for Mean-Field computation and firing rate measure
        # Excitatory
        d.spikes_mod[:, tsk] = 1 * d.matrix[:, 2]  # We store the spikes
        d.spikes[:, tsyp] = 1 * d.spikes_mod[:, tskp]
        # If we are just obtaining the initial conditions (a steady state) we don't need to
        # compute the firing rate.
        # Voltage measure:

        # ######################## -- FIRING RATE MEASURE -- ##
        fr.frspikes[:, tstep % fr.wsteps] = 1 * d.spikes[:, tsyp]
        fr.firingrate(tstep)
        # Distribution of Firing Rates
        if tstep > 0:
            fr.tspikes2 += d.matrix[:, 2]
            fr.ravg2 += 1  # Counter for the "instantaneous" distribution
            fr.ravg += 1  # Counter for the "total time average" distribution

    if d.sys in ('nf', 'both'):
        d.r[kp] = d.r[k] + d.dt / d.tau * (d.delta / pi + 2.0 * d.r[k] * d.v[k] - d.g * d.r[k] * d.r[k])
        d.v[kp] = d.v[k] + d.dt / d.tau * (
            d.v[k] ** 2 + d.eta0 - pi2 * d.r[k] ** 2 - d.g * d.r[k] * (d.v[k] - d.reverv))
        # if RuntimeWarning:
        #     logger.error("Overflow!")
            # break

    pbar.update(10 * tstep + 1)
    temps += d.dt
    tstep += 1

# Finish pbar
pbar.finish()
temps -= d.dt
tstep -= 1
# Stop the timer
print 'Total time: {}.'.format(timer() - time1)

# d.save(tsyp)
if args.pl:
    t = np.array(fr.tempsfr) * d.faketau
    r = np.array(fr.r) / d.faketau
    plt.plot(t, r)
    plt.plot(d.tpoints[0:tstep:10] * d.faketau, d.r[0:tstep:10] / d.faketau)
    plt.ylim([0, 100])
    plt.show()
