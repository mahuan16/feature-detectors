import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import process
import os
import pandas as pd
from tqdm import tqdm # loading bar for human eyes
from plot import draw_graph

# draw only keypoints location,not size and orientation
# drawn = cv.drawKeypoints(img1, kp, None, color=(0,255,0), flags=0)
# plt.imshow(drawn), plt.show()

NUM_KEYPOINTS = 1000
THRESHOLD = 3
PATH = "hpatches-sequences-release"
RESULTS_FILE = "results.csv"
NUM_FOLDERS = 60 # number of folders to iterate throug
# note that there are totla 59 viewpoint folders and total 57 illumination folders

def get_image_folders() -> list[str]:
    """
    read the name (it returns a list of the sequence folder names)
    """
    return sorted(os.listdir(PATH))

def main():
    # get v_ and i_ folder names
    
    viewpoint, illumination = [], []
    for scene in get_image_folders():
        if scene.startswith("v"):
            viewpoint.append(scene)
        else:
            illumination.append(scene)
    print(f"viewpoint: {len(viewpoint)}")
    print(f"illumination: {len(illumination)}")

    brisk = process.get_detector('brisk', NUM_KEYPOINTS)
    orb = process.get_detector('orb', NUM_KEYPOINTS)
    sift = process.get_detector('sift', NUM_KEYPOINTS)

    detectors = {"brisk": brisk, "orb": orb, "sift": sift}

    write_header = not os.path.exists(RESULTS_FILE)

    
    for name, d in detectors.items():
        for v in tqdm(viewpoint[:NUM_FOLDERS], desc=f"Viewpoint {name.upper()}",
                    colour="#90cfe0", ascii=" ▖▘▝▗▚▞█", ncols=100):
            results = one_folder(v, d, name)
            df = pd.DataFrame(results)
            df.to_csv(RESULTS_FILE, mode="a",
                    header=write_header, index=False)
            write_header = False

        for i in tqdm(illumination[:NUM_FOLDERS], desc=f"Illumination {name.upper()}",
                    colour="#f0d281", ascii=" ▖▘▝▗▚▞█", ncols=100):
            results = one_folder(i, d, name)
            df = pd.DataFrame(results)
            df.to_csv(RESULTS_FILE, mode="a",
                    header=write_header, index=False)
            write_header = False
    draw_graph(RESULTS_FILE)


    

def one_folder(current: str, detector, detector_name: str) -> list[dict]:
    path = os.path.join(PATH, current)
    img1 = process.load_gray(path, 1)
    kp, des, numkp, avgtime = process.detect(img1, detector, NUM_KEYPOINTS)
    kps_1 = numkp/avgtime if avgtime != 0 else np.nan
    # current_dict = {"repeatability": np.nan, "loc_error": np.nan, "visible_kp": np.nan, "repeated_kp": np.nan, "avg_time": avgtime, "num_kp": numkp}
    current_dict = {"scene": current, "detector": detector_name, "pair": 1, "type":current[0], "repeatability": np.nan, "loc_error": np.nan, "visible_kp": np.nan, "repeated_kp": np.nan, "avg_time": avgtime, "num_kp": numkp, "kps": kps_1}
    scenes = []
    scenes.append(current_dict)
    for i in range(2,7):
        imgN = process.load_gray(path, i)
        hN = process.load_homography(path, i)
        kpN, desN, numkpN, avgtimeN = process.detect(imgN, detector, NUM_KEYPOINTS)
        repeat, loc_error, viskp, repeated_kp = process.repeat_and_error(kp, kpN, hN, imgN.shape, THRESHOLD)

        kps = numkpN/avgtimeN if avgtimeN != 0 else np.nan
        # store metrics
        current_dict = {"scene": current, "detector": detector_name, "pair": i, "type":current[0], "repeatability": repeat, "loc_error": loc_error, "visible_kp": viskp, "repeated_kp": repeated_kp, "avg_time": avgtimeN, "num_kp": numkpN, "kps": kps}
        scenes.append(current_dict)

    return scenes

if __name__ == "__main__":
    main()