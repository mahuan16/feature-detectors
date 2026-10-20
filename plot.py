import os

import matplotlib.pyplot as plt
import pandas as pd

PLOTS_DIR = "plots"


def save_fig(fig, name):
   
    os.makedirs(PLOTS_DIR, exist_ok=True)  # make the folder if it doesn't exist
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"{name}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)  # free memory so charts can't draw on top of each other


def plot_repeat_by_type(df):
    means = df.groupby(["detector", "type"])["repeatability"].mean().unstack()

    fig, ax = plt.subplots(figsize=(7, 5))
    means.plot(kind="bar", ax=ax)
    ax.set_xlabel("Detector")
    ax.set_ylabel("Mean repeatability")
    ax.set_title("Repeatability by detector and scene type")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=0)
    save_fig(fig, "repeatability_by_type")


def plot_repeat_vs_pair(df):
    
    repeat = df.groupby(["detector", "pair"])["repeatability"].mean().unstack(0)

    fig, ax = plt.subplots(figsize=(7, 5))
    repeat.plot(kind="line", marker="o", ax=ax)
    ax.set_xlabel("Image pair (larger = bigger change)")
    ax.set_ylabel("Mean repeatability")
    ax.set_title("Mean repeatability vs pair number")
    ax.set_xticks(repeat.index)
    ax.set_ylim(0, 1)
    save_fig(fig, "repeatability_vs_pair")


def plot_keypoint_counts(df):
    counts = df.groupby("detector")[["num_kp", "visible_kp"]].mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set_xlabel("Detector")
    ax.set_ylabel("Number of keypoints")
    ax.set_title("Detected vs visible keypoints per detector")
    ax.tick_params(axis="x", rotation=0)
    save_fig(fig, "keypoint_counts")


def plot_loc_error(df):
    error = df.groupby(["detector", "type"])["loc_error"].mean().unstack()

    fig, ax = plt.subplots(figsize=(7, 5))
    error.plot(kind="bar", ax=ax)
    ax.set_xlabel("Detector")
    ax.set_ylabel("Mean localization error (pixels)")
    ax.set_title("Localization error by detector")
    ax.tick_params(axis="x", rotation=0)
    save_fig(fig, "loc_error")


def plot_detection_time(df):
    speed = df.groupby("detector")["avg_time"].mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    speed.plot(kind="bar", ax=ax)
    ax.set_xlabel("Detector")
    ax.set_ylabel("Average detection time (s)")
    ax.set_title("Average detection time")
    ax.tick_params(axis="x", rotation=0)
    save_fig(fig, "detection_time")


def plot_kp_per_sec(df):
    kps = df.groupby("detector")["kps"].mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    kps.plot(kind="bar", ax=ax)
    ax.set_xlabel("Detector")
    ax.set_ylabel("Average keypoints per second")
    ax.set_title("Keypoints per second")
    ax.tick_params(axis="x", rotation=0)
    save_fig(fig, "keypoints_per_second")


def plot_repeat_vs_speed(df):
    # scatter, not line: each dot is one row (speed, repeatability)
    clean = df.dropna(subset=["kps", "repeatability"])

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, group in clean.groupby("detector"):
        ax.scatter(group["kps"], group["repeatability"], label=name, alpha=0.6)
    ax.set_xscale("log")  # speeds can differ by a lot between detectors
    ax.set_xlabel("Keypoints per second (log scale)")
    ax.set_ylabel("Repeatability")
    ax.set_title("Repeatability vs speed")
    ax.set_ylim(0, 1)
    ax.legend(title="Detector")
    save_fig(fig, "repeatability_vs_speed")


def plot_repeat_spread(df):
    fig, ax = plt.subplots(figsize=(7, 5))
    df.boxplot(column="repeatability", by="detector", ax=ax)
    fig.suptitle("")  # pandas adds an automatic super-title when you use by=
    ax.set_xlabel("Detector")
    ax.set_ylabel("Repeatability")
    ax.set_title("Repeatability spread")
    ax.set_ylim(0, 1)
    save_fig(fig, "repeatability_spread")

def plot_avg_matches(df):
    avg = df.groupby("detector")["num_matches"].mean()

    fig, ax = plt.subplots(figsize=(7, 5))
    avg.plot(kind="bar", ax=ax)
    ax.set_title("Average Matches per Detector")
    ax.set_xlabel("Detector")
    ax.set_ylabel("Average Number of Matches")
    ax.tick_params(axis="x", rotation=0)

    save_fig(fig, "average_matches")

def plot_precision_recall(df):
    bars = df.groupby("detector")[["precision", "recall"]].mean()
    
    fig, ax = plt.subplots(figsize=(7, 5))
    bars.plot(kind="bar", ax=ax)
    ax.set_xlabel("Detector")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Average Precision and Recall")
    ax.set_title("Precision and Recall by Detector")
    ax.tick_params(axis="x", rotation=0)
    save_fig(fig, "precision_and_recall")

def plot_precision_recall_type(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)


    labels = [("v", "Viewpoint"), ("i", "Illumination")]

    for ax, (label, name) in zip(axes, labels):

        bars = df[df["type"] == label].groupby("detector")[["precision", "recall"]].mean()
        bars.plot(kind="bar", ax=ax)
        ax.set_ylabel("Average Precision and Recall")
        ax.set_xlabel("Detector")
        ax.set_ylim(0,1)
        ax.set_title(f"Precision and Recall for {name} Changes")

    save_fig(fig, "precision_and_recall_type")





def draw_graph(results_file):
    df = pd.read_csv(results_file)

    plot_repeat_by_type(df)
    plot_repeat_vs_pair(df)
    plot_keypoint_counts(df)
    plot_loc_error(df)
    plot_detection_time(df)
    plot_kp_per_sec(df)
    plot_repeat_vs_speed(df)
    plot_repeat_spread(df)

    plot_avg_matches(df)
    plot_precision_recall(df)
    plot_precision_recall_type(df)

    print(f"Saved plots to ./{PLOTS_DIR}/")


if __name__ == "__main__":
    draw_graph("results.csv")