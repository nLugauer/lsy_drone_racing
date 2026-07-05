"""Experiment logging and plotting utilities for lsy_drone_racing.

Runtime logging (``logging_utils``, ``sim_logging``) is dependency-free and
safe to import inside the simulation loop. The plotting modules
(``plotting``, ``plot_style``, ``make_all_plots``) additionally require
matplotlib / pandas / numpy and are intended to be run standalone from this
directory, e.g.::

    cd analysis
    python make_all_plots.py --results ../results --out ../figures
"""
