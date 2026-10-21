**How do SIFT, ORB, and BRISK compare in repeatability, localization error, and speed under viewpoint vs. illumination change?**

# Testing Different Feature Detectors
- SIFT (Scale-Invariant Feature Transform)
- ORB (Oriented FAST and Rotated BRIEF)
- BRISK (Binary Robust Invariant Scalable Keypoints)

We will be tracking a few metrics:
- Repeatability Rate
- Localization Error
- Keypoints Per Second
- Number of Matches
- Precision
- Recall

Across image sequences with viewpoint changes, and then across image sequences with illumination differences.

## Key Definitions
**Keypoint:** key points in an image, such as corners, edges, or blobs\
**Descriptor:** a unique signature for a keypoint (in feature detectors, descriptors are stored as an array of numbers); includes information on pixel intensities, edge directions, gradients; ideally, a good descriptor should remain invariant to transformations like rotation, scaling, changes in illumination/viewpoint\
**SIFT:** Scale-Invariant Feature Transform; developed in 1999, and specializes in scale and rotation changes (good at identifying same keypoints and descriptors even when you zoom in on an object or rotate the image)\
**ORB:** Oriented FAST and Rotated BRIEF; made in 2011, it specializes in speed, and the fact that it is open-source alternative to patented algorithms like SIFT and SURF. It is faster because it uses integers and logical operations rather than float math. It is a combination of FAST (used for finding corners) and BRIEF (used to convert those points into binary data vectors for easy comparison)\
**BRISK:** Binary Robust Invariant Scalable Keypoints;  uses AGAST (a FAST variant) to find corners and keypoints across different scales. Its descriptors are binary and made through comparing brightnesses between points around the keypoint\
**Homography:** A 3x3 transformation matrix that maps an image (a flat scene) onto another plane (basically the image rotated)\
**Repeatability Rate:** fraction of how many keypoints from the original image were detected after being projected into the second image\
**Keypoints Per Second:** the number of keypoints detected (before trimming) divided by the average detect and compute time on an image in seconds\
**Match:** a descriptor pair of a descriptor from the original image, and the nearest descriptor from the image projection after passing the ratio test (the two nearest descriptors are found, and if the closer one is less than or equal to 0.75 of the further one, then the closer is kept as a match)
**Precision:**refers to how many of the matches detected were accurate, basically if the point in image two is within 3px of the corresponding point in the projection of image 1 (true positive / total matches found)\
**Recall:**refers to how many descriptors from the original image actually had an accurate match in the transformed image (true positive / repeated keypoints).\

<ul style="margin-top: -10px; margin-bottom: 0px; padding-left: 20px;">
  <li style="margin-bottom: 2px;">Note: we only check the original image's projected points onto image N, not the other way around</li>
</ul>

**Localization Error:** average distance (pixels) between the projection of image 1's keypoints and the keypoints in the second image

## Setup and Limitations
The dataset used was from HPatches Sequences (https://github.com/hpatches/hpatches-dataset/tree/master)

Each detector was allowed 1000 keypoints, and the keypoints were then trimmed to the top 1000 strongest keypoints (for BRISK, a keypoint cap couldn't be set so it tends to have many more keypoints trimmed compared to SIFT and ORB). The pixel threshold was 3, and the ratio test value (for checking if the keypoints matched the descriptors) was 0.75. Each detector was run on an image three times, and the average time was taken for the kps (note that an initial run was done for all three as the first run can often be slower to build cache; only the three after the first run were used for calculating average).

There were a total of 59 viewpoint scenes and 57 illumination scenes.

### Device Specifications
This experiment was run on an HP Envy x360
- CPU: AMD Ryzen 7 8840HS
    - Note that everything was run on the CPU, no GPU acceleration
- RAM: 16 GB
- OS: Windows 11 Home, version 25H2

Code was run while laptop was on battery (above 50%, low power mode). No other heavy background processes were open during the timed runs. Note that the timed runs are more so meant to compare to one another, not as a standalone speed metric.

### Tech Stack
- Python 3.13.4
- OpenCV 4.14.0.94
    - Note: Using opencv-python 4.x as BRISK moved to a different package after ver 5+
- NumPy 2.5.3
- Matplotlib 3.11.2
- Pandas 3.0.6
- Scipy 1.18.1
- tqdm 4.70.1

## Results
### Overall Findings
- ORB was overall the fastest and most localized detector.
    - ~0.02s per image, roughly 5x faster than SIFT and 7x faster than BRISK
    - lowest average localization error
    - However, also had the widest spread (least predictable)
- SIFT was the slowest, but most consistent performance
    - Good at finding correct matches, but had the worst repeatability
- BRISK had the best preicision and decent repetability
    - Technically slowest, but likely due to the fact it cannot cap its keypoints before computing

**Repeatable != Matchable**
ORB had the highest repeatability but lowest recall, whereas SIFT had the lowest repeatability but highest recall and most matches

**Speed Doesn't Predict Quality**
Each detector tended to have an average detection speed, but a wide range of repeatability.

### Detection
<img src="plots_iter2/keypoint_counts.png" style="max-width: 650px;" alt="Bar chart displaying average number of keypoints each detector found">

The visible keypoints count tends to be lower than the original number of generated keypoints. For BRISK, a set keypoint count couldn't be passed in so the original amount was very high and got trimmed to 1000 later which is why there is such a sizeable gap. All three detectors lost some keypoints after translation as they were outside of the projected image, so the visible kp is slightly lower than the originally generated ones.

### Repeatability and Localization
To clarify the definition of repeatability, it is the number of keypoints from the original image that were also located in the transformed image. Note that this experiment uses a simplified definition of repeatability, as usually it is also tested by mapping the transformed keypoints back onto the original image./
Localization error is measured in pixels, and describes the distance the transformed keypoint is from the one in the original image.

<img src="plots_iter2/repeatability_by_type.png" style="max-width: 650px;" alt="Bar chart displaying repeatability for each detector on viewpoint and illumination changes">

All three detectors tended to have a higher average repeatability when testing against viewpoint changes. ORB had the highest repeatability for both illumination and viewpoint changes, but no detector seemed to surpass 70% repeatability (note that this number is likely influenced by parts of the methodology). SIFT had the lowest mean repeatability for both viewpoint and illumination changes.

<img src="plots_iter2/repeatability_vs_pair.png" style="max-width: 650px;" alt="Line chart displaying detector repeatability as image pair increases (larger image pair means bigger transformation)">

As the image pair increases (larger viewpoint image pair means a more extreme transformation), all three detectors performed worse and had a lower repeatability. ORB still demonstrates the highest average repeatability for each image pair (between 60%-75%), with BRISK exhibiting similar but slightly poorer performance (between 60-70%). SIFT is much worse, with its highest performance at the 50% repeatability range.

<img src="plots_iter2/repeatability_spread.png" style="max-width: 650px;" alt="Box chart displaying repeatability spread for each detector">

ORB has the highest median repeatability, and also the widest IQR. SIFT has the smallest IQR, but also doesn't near 100% repeatability with its whiskers. BRISK has a median repeatability near 60%, and also has a decently large IQR with its whiskers nearing 97%.

<img src="plots_iter2/loc_error.png" style="max-width: 650px;" alt="Bar chart displaying localization error in pixels for each detector on viewpoint and illumination changes">

The localization error for ORB is the lowest for both viewpoint and illumination changes (note that a threshold of 3px was used). All three detectors have similar localization errors for viewpoint changes, around 1.2 pixels. However, ORB had a noticeably lower localization error for illumination changes (around 0.88) whereas SIFT and BRISK had errors closer to 1.1 pixels.


### Matching
To find matches, the two nearest neighbors of a descriptor are found. Then, the distances are compared, and if the best (closer) distance is less than 0.75 of the second best neighbor's distance (meaning there is a sizeable gap between the two), then the best neighbor is kept as the match. This is designed to reject ambiguous matches where there isn't a clear "best distance."

<img src="plots_iter2/average_matches.png" style="max-width: 650px;" alt="Bar chart displaying average number of matching descriptors after image transformation">

SIFT had the largest mean matches, exceeding 200, whereas BRISK and ORB both hovered around 160-170 mean matches.

<img src="plots_iter2/precision_and_recall.png" style="max-width: 650px;" alt="Bar chart displaying average precision and recall for each detector">

BRISK had the highest precision, around 0.8, with SIFT and ORB close behind. However, all three detectors struggled considerably more with recall (note that this is partially due to the 0.75 ratio test which may have discarded correctly recalled points). The highest recall was from SIFT (around 0.4), whereas BRISK was around 0.25 and ORB was around 0.2.

<img src="plots_iter2/precision_and_recall_type.png" style="max-width: 650px;" alt="Bar chart displaying average precision and recall for each detector on illumination and viewpoint changes">

Each detector had similar precision and recall percentages through both viewpoint and illumination changes.

### Speed and Tradeoff
<img src="plots_iter2/keypoints_per_second.png" style="max-width: 650px;" alt="Bar chart displaying a detector's keypoints per second">

ORB and BRISK were the fastest, while SIFT was the slowest.

<img src="plots_iter2/detection_time.png" style="max-width: 650px;" alt="Bar chart displaying average detector time in seconds">

ORB has the fastest average detection time (around 0.02 s per image) and SIFT comes second (with around 0.10 s per image). BRISK has an average detection time of around 0.14 s per image, but it should be noted that BRISK did not have a capped keypoints parameter during keypoint detection (the keypoints were trimmed to the top 1000 after generation), which likely inflated its average detection time.

<img src="plots_iter2/repeatability_vs_speed.png" style="max-width: 650px;" alt="Scatter plot displaying repeatability as keypoints per second increases">

Despite the detectors having consistent detection speeds (keypoints per second), their repeatability is very varied. Speed doesn't have much of an impact on repeatability.


## For the Future
Compare Against: ALGD-ORB
- Trying Adaptive Threshold and Local Gray Difference-ORB (ALGD-ORB)

Compare Against: Personally Created Feature Detector

Measure More Metrics:
- Keypoint Distribution
- Test With Actual Visual Odometry
- Different Thresholds

# Sources
- credit to https://github.com/hpatches/hpatches-dataset/tree/master for sequence images