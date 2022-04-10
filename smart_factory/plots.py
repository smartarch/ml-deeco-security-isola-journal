import matplotlib
import matplotlib.pyplot as plt
import numpy as np

font = {'size': 12}

matplotlib.rc('font', **font)


def createPlot(shiftsLog, iterations, simulations, filename, show=False, figsize=None):

    if not figsize:
        figsize = (10, 10)
    fig, ax_s = plt.subplots(figsize=figsize)

    xLabels = ["M", "T", "W", "T", "F", "S", "S"][:simulations] * iterations
    x = list(range(1, iterations * simulations + 1))

    standbys = shiftsLog.getColumnAvg("standbys")
    lateness = shiftsLog.getColumnAvg("lateness")

    ax_l = ax_s.twinx()

    ax_s.plot(x, standbys, c='tab:blue')
    ax_l.plot(x, lateness, c='tab:orange')

    if iterations > 1:
        yLines = np.linspace(simulations + 0.5, ((iterations - 1) * simulations) + 0.5, iterations - 1)
        ax_s.vlines(x=yLines, colors='black', ymin=0, ymax=max(standbys), linestyle='dotted')
        twin_y = ax_s.twiny()
        twin_y.set_xticks(yLines, labels=[f"Training {i + 1}" for i in range(iterations - 1)])

    ax_s.set_xticks(x, labels=xLabels)

    ax_s.set_ylabel("Standbys")
    ax_l.set_ylabel("Lateness")

    fig.tight_layout()
    plt.savefig(filename)
    if show:
        plt.show()
    plt.close(fig)
