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


# def process(img_path, img_num, detector, threshold, max_n):

#     og, img, h = load_image_pair(img_path, img_num)

#     start = time.perf_counter()

#     kp, des = detector.detectAndCompute(og, None)
#     num_kp = len(kp)
#     kp, des = trim_output(kp, des, max_n)

#     kpN, desN = detector.detectAndCompute(img, None)
#     num_kpN = len(kpN)
#     kpN, desN = trim_output(kpN, desN, max_n)

#     end = time.perf_counter()
#     elapsed = end-start # time taken

#     result = repeat_and_error(kp, kpN, h, og.shape, threshold)

#     return elapsed, result, (num_kp, num_kpN)




# ------------- testing the repeat and error func --------------


# path = r"hpatches-sequences-release\v_adam"

# brisk = get_detector("brisk", 1000)

# og, img, h = load_image_pair(path, 2)
# kp, des = brisk.detectAndCompute(og, None)
# kpN, desN = brisk.detectAndCompute(img, None)

# print(f"brisk 5: {repeat_and_error(kp, kpN, h, og.shape, 5)}")
# print(f"brisk 3: {repeat_and_error(kp, kpN, h, og.shape, 3)}")
# print(f"brisk 1: {repeat_and_error(kp, kpN, h, og.shape, 1)}")


# thing = get_detector("brisk", 1000)
# og, img = cv.imread("img/sample1.jpg", cv.IMREAD_GRAYSCALE), cv.imread("img/sample1.jpg", cv.IMREAD_GRAYSCALE)
# h = np.eye(3)

# kp, des = thing.detectAndCompute(og, None)
# kp, des = trim_output(kp, des, 1000)

# kpN, desN = thing.detectAndCompute(img, None)
# kpN, desN = trim_output(kp, des, 1000)

# print(repeat_and_error(kp, kpN, h, og.shape, 3))

# ==========
# h[0][2] = 5 # shift right by 5
# print(f"Toy Shift: {repeat_and_error(kp, kpN, h, (1000, 1000), 3)}")

# kpN = [cv.KeyPoint(16, 10, 1), cv.KeyPoint(56, 50, 1)]
# print(f"Toy Offset: {repeat_and_error(kp, kpN, h, (1000,1000), 3)}")

# kpN = [cv.KeyPoint(15, 10, 1), cv.KeyPoint(55, 50, 1), cv.KeyPoint(1001, 1000, 1)]
# print(f"Toy Offset: {repeat_and_error(kp, kpN, h, (1000,1000), 3)}")

# print(f"empty: {repeat_and_error([], [], h, (100, 100), 3)}")