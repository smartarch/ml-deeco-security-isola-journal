import matplotlib.pyplot as plt
import numpy as np


def plotFailureRate(machineLogs, maxMachines=None, filename=None, show=False, figsize=None):
    if not figsize:
        figsize = (9, 5)
    machines = list(machineLogs.keys())
    if maxMachines is not None:
        machines = machines[:maxMachines]
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(machineLogs[machines[0]].records))

    for machine in machines:
        y = [d[3] for d in machineLogs[machine].records]
        plt.plot(x, y, label=machine.id)

    ax.set_xlabel("Time")
    ax.set_ylabel("Failure rate")
    ax.legend()

    plt.title("Failure rate of machines")

    fig.tight_layout()
    if filename:
        plt.savefig(filename, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
