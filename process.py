import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from collections.abc import Sequence
from scipy import spatial
import time

NUM_LOOPS = 3 # the number of times to detectandcompute so we can calculate average time
"""the number of times to detectandcompute so we can calculate average time"""

def load_image_pair(folder_path: str, num: int):
    """
    Takes in img_path and returns a loaded, 
    grayscale version of the image and your desired pair.
    \nI love you maria you're so cool :3
    """
    og_img = os.path.join(folder_path, "1.ppm")
    comp_img = os.path.join(folder_path, f"{num}.ppm")
    homography = os.path.join(folder_path, f"H_1_{num}")

    return cv.imread(og_img, cv.IMREAD_GRAYSCALE), cv.imread(comp_img, cv.IMREAD_GRAYSCALE), np.loadtxt(homography)

def load_gray(path: str, num: int = 1):
    """
    returns grayscale version of image; defaults to image 1
    """
    img_path = os.path.join(path, f"{num}.ppm")
    return cv.imread(img_path, cv.IMREAD_GRAYSCALE)

def load_homography(path: str, num: int):
    """
    returns image num's homography
    """
    h_path = os.path.join(path, f"H_1_{num}")
    return np.loadtxt(h_path)

def get_detector(name: str, n_cap: int):
    """
    Takes in the name of the detector (ORB, SIFT, or BRISK), 
    and also the max for nfeatures, and returns it
    """
    name = name.upper() # in case someone wrote orb instead of ORB
    feature_dict = {"ORB": lambda n: cv.ORB.create(nfeatures=n),
                    "SIFT": lambda n: cv.SIFT.create(nfeatures=n),
                    "BRISK": lambda n: cv.BRISK.create()} 
    if name not in feature_dict:
        raise ValueError(f"{name!r} not in dictionary, valid options: {list(feature_dict)}")

    return feature_dict[name](n_cap)

def trim_output(keypoints: Sequence[cv.KeyPoint], descriptors: np.ndarray | None, n_cap: int) -> tuple[Sequence[cv.KeyPoint], np.ndarray | None]:
    """
    Accepts keypoints, descriptors, and nfeatures cap, 
    then trims the keypoints so it is the top n_cap 
    strongest keypoints+descriptors
    """

    if not keypoints or descriptors is None:
        return keypoints, descriptors

    response_list = [k.response for k in keypoints]

    # sort the indices in res list from strongest response to weakest,
    # then only keep the n_cap strongest ones
    sorted_res_i = np.argsort(response_list, descending=True)
    sorted_res_i = sorted_res_i[:n_cap]

    trimmed_list = [keypoints[i] for i in sorted_res_i]
    trimmed_desc = descriptors[sorted_res_i]

    return trimmed_list, trimmed_desc

def repeat_and_error(og_kp, img_kp, h, img_size: tuple[int, int], threshold) -> tuple[int | float, int | float, int | float, int | float]:
    """
    Note that img_size should be from img.shape which returns a (height, width) kinda thing \n
    out put is repeatability, localization error, number of visible points in transformed image, and number of points that were repeated
    """
    if not og_kp or not img_kp:
        return np.nan, np.nan, 0, 0

    og_pts = [kp.pt for kp in og_kp]
    og_pts = np.array(og_pts, dtype=np.float32) # make sure they are float32 otherwise possible errors
    img_pts = [kp.pt for kp in img_kp]
    img_pts = np.array(img_pts, dtype=np.float32)

    reshape_pts = og_pts.reshape(-1, 1, 2)
    transformed = cv.perspectiveTransform(reshape_pts, h)

    transformed = transformed.reshape(-1, 2) # transformed the original image keypoints onto
                                             # img_pts's coord system

    x_col = transformed[:,0]
    y_col = transformed[:, 1]

    x_arr = (x_col >= 0) & (x_col < img_size[1])
    y_arr = (y_col >= 0) & (y_col < img_size[0])

    visible = (x_arr) & (y_arr) # bool array mask to see which kp on og image are
                                # actually visible on image_pts coord sys

    kp_tree = spatial.KDTree(img_pts)
    visible_pts = transformed[visible]

    if len(visible_pts) <= 0: # no visible points 
        return np.nan, np.nan, len(visible_pts), np.nan
    dist, kp_i = kp_tree.query(visible_pts) # built tree with image pts, now checking
                                            # dist between kp on img and og img (transformed ver)

    # calculate the keypoints that were detected again in the transformed version
    # if the distance of the new kp is less than a certain threshold comarped to old kp,
    # then we say it's repeated
    repeat_bool = dist < threshold # this is an arr btw (built automatically w numpy)
    if int(np.sum(repeat_bool)) <= 0:
        return 0, np.nan, len(visible_pts), int(np.sum(repeat_bool))

    repeatability = float(np.mean(repeat_bool)) # how many points 
    loc_error = float(np.mean(dist[repeat_bool]))

    return repeatability, loc_error, len(visible_pts), int(np.sum(repeat_bool))

def detect(img, detector, max_n: int) -> tuple[Sequence[cv.KeyPoint], np.ndarray | None, int, float]:
    """
    detects and computes on given image with given detector, 
    returns the trimmed keypoints and descriptors, 
    as well as the length of the original keypoints and the average detectandcompute time
    """

    kp, des = detector.detectAndCompute(img, None) # warm up (builds initial cache)

    average = []
    for i in range(NUM_LOOPS):
        start = time.perf_counter()
        kp, des = detector.detectAndCompute(img, None)
        end = time.perf_counter()
        average.append(end-start)

    new_kp, new_des = trim_output(kp, des, max_n)
    return new_kp, new_des, len(kp),  np.median(average)

def matching(des, desN, kp, kpN, matcher, h, threshold: float | int, num_repeated, r: float) -> tuple[ float,  float, int, int ]:
    """
    match precision, match recall, the number of matches after the ratio test, 
    and the number of correct matches.
    """
    if des is None or desN is None:
        return np.nan, np.nan, 0, 0

    nearest = matcher.knnMatch(des, desN, 2)

    matches = []
    for item in nearest:
        if len(item) < 2:
            continue
        first, sec = item
        if first.distance <= r*sec.distance:
            matches.append(first)

    original = [kp[m.queryIdx].pt for m in matches]
    matched_pts = [kpN[m.trainIdx].pt for m in matches]
    
    original = np.array(original, dtype=np.float32)

    if not original.any(): # i hate njmpy arrays.........................
        return np.nan, np.nan, len(matches), 0

    reshape_pts = original.reshape(-1, 1, 2)
    transformed = cv.perspectiveTransform(reshape_pts, h)
    transformed = transformed.reshape(-1, 2) # what the matches should be looking like

    diff = matched_pts - transformed

    dist = np.linalg.norm(diff, axis=1)

    dist_bool = dist < threshold # boolean array i swear

    correct = np.sum(dist_bool)

    precision = correct/len(matches) if len(matches) else np.nan
    recall = correct/num_repeated if num_repeated > 0 else 0
        

    return float(precision), float(recall), len(matches), int(correct)


# orb = get_detector("orb", 1000)

# path = "hpatches-sequences-release/v_abstract"
# mine = load_gray(path)
# h = np.eye(3)
# h[0][2] = 50

# kp, des, kp_num, time = detect(mine, orb, 1000)

# repeat, loc_error, vis_n, repeat_n = repeat_and_error(kp, kp, h, mine.shape, 3)

# orb_m = cv.BFMatcher(cv.NORM_HAMMING)
# print(matching(des, des, kp, kp, orb_m, h, 3, repeat_n, 0.75))

