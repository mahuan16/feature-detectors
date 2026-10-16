from matplotlib import pyplot as plt
import pandas as pd



def draw_graph(results_file):
    df = pd.read_csv(results_file)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    
    
    # average repeatability
    means = df.groupby(["detector", "type"])["repeatability"].mean()
    means = means.unstack()
    means.plot(kind="bar", ax=axes[0, 0])
    axes[0, 0].set_ylabel("Mean repeatability (kp/kp)")
    axes[0, 0].set_title("Repeatability by detector and scene type")


    # Repeatability vs pair number
    repeat = df.groupby(["detector", "pair"])["repeatability"].mean().unstack(0)
    repeat.plot(kind="line", ax=axes[0,1])
    axes[0,1].set_ylabel("Mean Repeatability")
    axes[0,1].set_title("Mean Repeatability vs Pair Number")
    axes[0,1].set_xticks(repeat.index)
    axes[0,1].set_ylim(0, 1)

    # keypoints and visible kps per detector
        
    num_kp = df.groupby(["detector"])[["num_kp", "visible_kp"]].mean()
    num_kp.plot(kind="bar", ax=axes[0, 2])
    axes[0, 2].set_ylabel("Number of Keypoints")
    axes[0, 2].set_title("Num Keypoints vs Visible Keypoints per Detector")


    # loc error by detector
    # plot the average loc error per detector per pair?
    error = df.groupby(["detector", "type"])["loc_error"].mean().unstack()
    error.plot(kind="bar", ax=axes[1, 0])
    axes[1, 0].set_ylabel("Mean Localization Error (pixels?)")
    axes[1, 0].set_title("Localization Error by Detector")
  

    # speed by detector
    speed = df.groupby(["detector"])["avg_time"].mean()
    speed.plot(kind="bar", ax=axes[1, 1])
    axes[1, 1].set_ylabel("Average Detection Speed (s)")
    axes[1, 1].set_title("Average Detection Time")

    kps = df.groupby(["detector"])["kps"].mean()
    kps.plot(kind="bar", ax=axes[1, 2])
    axes[1, 2].set_ylabel("Average Keypoints per Second (kp/s)")
    axes[1, 2].set_title("Keypoints per Second")

    plt.tight_layout(pad=2.0)
    
    plt.show()

    # repeativility vs speed

    # repeativbility dist



