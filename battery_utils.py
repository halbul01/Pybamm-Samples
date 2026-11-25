"""Helper functions for analysing PyBaMM charging solutions.

Usage in a notebook (same folder):
    from battery_utils import step_table, charge_summary
"""
import numpy as np
import pandas as pd


def _soc(sol, Q_nom, initial_soc=0):
    """SoC [%] over time.

    initial_soc can be given as a fraction (0 ... 1) or in percent (0 ... 100),
    the same value you pass to parameter_values.set_initial_state().
    Discharge capacity grows when discharging and falls when charging.
    """
    if initial_soc > 1:          # given in percent, e.g. 100
        initial_soc = initial_soc / 100
    return (initial_soc - sol["Discharge capacity [A.h]"].entries / Q_nom) * 100


def step_table(sol, Q_nom, initial_soc=0, labels=None):
    """Duration and charged SoC for every experiment step (cycle).

    labels: optional list of step names, e.g. ["2C CC", "1C CC", "CV"].
    """
    if labels is None:
        labels = [f"Step {i + 1}" for i in range(len(sol.cycles))]

    rows = []
    for name, cyc in zip(labels, sol.cycles):
        t_c = cyc["Time [min]"].entries
        soc_c = _soc(cyc, Q_nom, initial_soc)
        rows.append([name, t_c[-1] - t_c[0], soc_c[-1] - soc_c[0], soc_c[-1]])

    # Delta SoC: positive = charged, negative = discharged
    columns = ["Step", "Duration [min]", "Delta SoC [%]", "SoC at end [%]"]
    return pd.DataFrame(rows, columns=columns).round(2)


def charge_summary(sol, Q_nom, initial_soc=0, t_check=10, soc_check=80):
    """Key numbers for one charge or discharge: SoC at t_check min,
    time to reach soc_check %, totals.

    Works in both directions: starting at 0 % and charging up to soc_check,
    or starting at 100 % and discharging down to soc_check.

    Returns a dict, so several runs can be collected into one DataFrame:
        pd.DataFrame([charge_summary(s, Q_nom) for s in sols.values()])
    """
    t = sol["Time [min]"].entries
    soc = _soc(sol, Q_nom, initial_soc)
    u = sol["Negative electrode surface potential difference at separator interface [V]"].entries * 1000  # mV

    # Time when SoC first crosses soc_check (NaN if never reached).
    # Works for charging (SoC rising) and discharging (SoC falling).
    charging = soc[-1] > soc[0]
    reached = (soc >= soc_check) if charging else (soc <= soc_check)
    if not reached.any():
        t_soc = np.nan
    else:
        i = np.argmax(reached)               # first index past soc_check
        if i == 0:
            t_soc = t[0]                     # already there at the start
        else:                                # linear interpolation between i-1 and i
            frac = (soc_check - soc[i - 1]) / (soc[i] - soc[i - 1])
            t_soc = t[i - 1] + frac * (t[i] - t[i - 1])

    # Total time with anode potential below 0 mV (lithium plating risk)
    dt = np.diff(t)
    t_below_0 = np.sum(dt[u[:-1] < 0])

    result = {
        f"SoC at {t_check} min [%]": np.interp(t_check, t, soc),
        f"Time to {soc_check}% SoC [min]": t_soc,
        "Total time [min]": t[-1],
        "Final SoC [%]": soc[-1],
        "Min anode potential [mV]": u.min(),
        "Time below 0 mV [min]": t_below_0,
    }
    # Plain Python floats, so the dict prints as 30.9 instead of np.float64(30.9)
    return {k: round(float(v), 2) for k, v in result.items()}
