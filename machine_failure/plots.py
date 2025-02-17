import itertools

import matplotlib.pyplot as plt
import numpy as np


def plotFailureRate(machineLogs, maxMachines=None, filename=None, show=False, figsize=None, title="Failure rate of machines"):
    if not figsize:
        figsize = (10, 4.5)
    machines = list(machineLogs.keys())
    if maxMachines is not None:
        machines = machines[:maxMachines]
    fig, [ax_fr, ax_run] = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])

    x = np.arange(len(machineLogs[machines[0]].records))

    for machine in machines:
        y = [d[3] for d in machineLogs[machine].records]
        ax_fr.plot(x, y, label=machine.id)

    ax_fr.set_xlim(0, len(x))
    # ax_fr.set_xlabel("Time")
    ax_fr.set_ylabel("Failure rate")
    # ax.legend()

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    for i, machine in enumerate(machines):
        ranges = []
        index = 0
        for val, group in itertools.groupby(d[2] for d in machineLogs[machine].records):
            group_len = len(list(group))
            if not val:
                ranges.append((index, group_len))
            index += group_len
        # color = colors[i]
        color = "tab:red"
        ax_run.broken_barh(ranges, (i + 1 - 0.4, 0.8), facecolors=color)
    ax_run.set_xlim(*ax_fr.get_xlim())
    ax_run.set_ylim(0.5, len(machines) + 0.5)
    if len(machines) == 1:
        ax_run.set_yticks([])
    else:
        ax_run.set_yticks(np.arange(len(machines)) + 1)

    ax_run.set_xlabel("Time")
    # ax_run.set_ylabel("Machine running")
    ax_run.set_ylabel("Machine\nnot running")

    fig.suptitle(title)

    fig.tight_layout(pad=0.35)
    if filename:
        plt.savefig(filename, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
