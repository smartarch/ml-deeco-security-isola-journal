import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from configuration import CONFIGURATION
from helpers import DayOfWeek
from ml_deeco.estimators import CategoricalFeature

font = {'size': 12}

matplotlib.rc('font', **font)


def plotStandbysAndLateness(shiftsLog, iterations, simulations, filename, show=False, figsize=None):
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


def plotLateWorkersNN(estimator, filename=None, figsize=None):
    timeSteps = CONFIGURATION.shiftStart + 1
    timeToShift = np.linspace(CONFIGURATION.shiftStart / CONFIGURATION.steps, 0, timeSteps)
    daysOfWeekFeature = CategoricalFeature(DayOfWeek)

    records = []
    for day in DayOfWeek:
        for time in timeToShift:
            record = np.concatenate([np.array([time]), daysOfWeekFeature.preprocess(day)])
            records.append(record)
    records = np.array(records)

    outputs = estimator.predictBatch(records)
    outputs = outputs.reshape([7, timeSteps])

    yTickLabels = ["M", "T", "W", "T", "F", "S", "S"]
    xTickLabels = [str(x) if x % 3 == 0 else "" for x in range(CONFIGURATION.shiftStart, -1, -1)]

    sns.heatmap(outputs, mask=outputs >= 0.5,
                vmin=0, vmax=1,
                cmap=sns.dark_palette("seagreen", as_cmap=True),
                yticklabels=yTickLabels, xticklabels=xTickLabels)
    sns.heatmap(outputs, mask=outputs < 0.5,
                vmin=0, vmax=1,
                cmap=sns.dark_palette("salmon", as_cmap=True),
                yticklabels=yTickLabels, xticklabels=xTickLabels)
    plt.show()

    if filename:
        plt.savefig(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Smart factory simulation NN plot.')
    parser.add_argument('-o', '--output_folder', type=str, help='Output folder of the simulation.', required=True, default='results')
    args = parser.parse_args()

    folder = Path(args.output_folder)
    import tensorflow as tf

    model = tf.keras.models.load_model(folder / "late_workers" / "model.h5")


    class EstimatorDummy:
        def predictBatch(self, x):
            return model(x).numpy()


    plotLateWorkersNN(EstimatorDummy())
