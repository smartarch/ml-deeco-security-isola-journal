import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from configuration import CONFIGURATION
from helpers import DayOfWeek
from ml_deeco.estimators import CategoricalFeature

font = {'size': 12}

matplotlib.rc('font', **font)


def computeWeeklyAverages(data, iterations, simulations):
    result = []
    for i in range(iterations):
        weekStart = i * simulations
        week = data[weekStart:weekStart + simulations]
        result.extend([mean(week[:5])] * 5)   # work days
        result.extend([mean(week[-2:])] * 2)  # weekend
    return result


def plotStandbysAndLateness(shiftsLog, iterations, simulations, filename=None, show=False, figsize=None):
    if not figsize:
        figsize = (10, 10)
    fig, ax_s = plt.subplots(figsize=figsize)

    xLabels = ["M", "T", "W", "T", "F", "S", "S"][:simulations] * iterations
    x = list(range(1, iterations * simulations + 1))

    standbys = shiftsLog.getColumnAvg("standbys")
    standbysAvg = computeWeeklyAverages(standbys, iterations, simulations)
    lateness = shiftsLog.getColumnAvg("lateness")
    latenessAvg = computeWeeklyAverages(lateness, iterations, simulations)

    ax_l = ax_s.twinx()

    legend = []
    legend.append(ax_s.plot(x, standbys, c='tab:blue', marker="o", linestyle="None", label="Standbys"))
    ax_s.plot(x, standbysAvg, c='tab:blue', linestyle="dashed")
    legend.append(ax_l.plot(x, lateness, c='tab:orange', marker="o", linestyle="None", label="Lateness"))
    ax_l.plot(x, latenessAvg, c='tab:orange', linestyle="dashed")

    if iterations > 1:
        yLines = np.linspace(simulations + 0.5, ((iterations - 1) * simulations) + 0.5, iterations - 1)
        ax_s.vlines(x=yLines, colors='black', ymin=0, ymax=max(standbys), linestyle='dotted')
        twin_y = ax_s.twiny()
        twin_y.set_xlim(ax_s.get_xlim())
        twin_y.set_xticks(yLines, labels=[f"Training {i + 1}" for i in range(iterations - 1)])

    ax_s.set_xticks(x, labels=xLabels)
    ax_s.set_xlabel("Day of week")

    ax_s.set_ylabel("Standbys")
    ax_l.set_ylabel("Lateness")

    lines = [line for lines in legend for line in lines]
    labels = [line.get_label() for line in lines]
    ax_l.legend(lines, labels)

    plt.title("Number of standbys and lateness in the smart factory simulation")

    fig.tight_layout()
    if filename:
        plt.savefig(filename)
    if show:
        plt.show()
    plt.close(fig)


def plotLateWorkersNN(estimator, filename=None, subtitle="", show=False, figsize=None):
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

    plt.xlabel("Time to shift")
    plt.ylabel("Day of week")
    title = "Neural network output"
    if subtitle:
        title += "\n" + subtitle
    plt.title(title)

    if filename:
        plt.gcf().savefig(filename)
    if show:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Smart factory simulation NN plot.')
    parser.add_argument('-o', '--output_folder', type=str, help='Output folder of the simulation.', required=True, default='results')
    args = parser.parse_args()

    folder = Path(args.output_folder)

    # standbys and lateness
    with open(folder / "shifts_avg.csv", newline="") as csvfile:
        reader = csv.reader(csvfile)
        data = list(reader)
    latenessIndex = data[0].index("lateness")
    standbysIndex = data[0].index("standbys")
    del data[0]

    class DummyShiftsLog:
        def getColumnAvg(self, col):
            if col == "lateness":
                return [float(d[latenessIndex]) for d in data]
            elif col == "standbys":
                return [float(d[standbysIndex]) for d in data]

    plotStandbysAndLateness(DummyShiftsLog(), len(data) // 7, 7, show=True)

    # NN
    import tensorflow as tf

    model = tf.keras.models.load_model(folder / "late_workers" / "model.h5")


    class EstimatorDummy:
        def predictBatch(self, x):
            return model(x).numpy()


    plotLateWorkersNN(EstimatorDummy(), show=True, filename=folder / "nn.png")
