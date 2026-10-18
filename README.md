**How do SIFT, ORB, and BRISK compare in repeatability, localization error, and speed under viewpoint vs. illumination change**

# Credits
- credit to https://github.com/hpatches/hpatches-dataset/tree/master for sequence images

# Testing Different Feature Detectors
- SIFT (Scale-Invariant Feature Transform)
- ORB (Oriented FAST and Rotated BRIEF)
- BRISK (Binary Robust Invariant Scalable Keypoints)

We will be tracking a few metrics:
- Repeatability Rate
- Localization Error
- Keypoints Per Second
Across image sequences with viewpoint changes, and then across image sequences with illumination differences.

## Key Definitions
**Keypoint:** key points in an image, such as corners, edges, or blobs
**Descriptor:** a unique signature for a keypoint (in feature detectors, descriptors are stored as an array of numbers); includes information on pixel intensities, edge directions, gradients; ideally, a good descriptor should remain invariant to transformations like rotation, scaling, changes in illumination/viewpoint
**SIFT:** Scale-Invariant Feature Transform; developed in 1999, and specialices in scale and rotation changes (good at identifying same keypoints and descriptors even when you zoom in on an object or rotate the image)
**ORB:** Oriented FAST and Rotated BRIEF; made in 2011, it specializes in speed, and the fact that it is open-source (faster because it uses integers and logical operations rather than float math). It is a combination of FAST (used for finding corners) and BRIEF (used to convert those points into binary data vectors for easy comparison)
**BRISK:** Binary Robust Invariant Scalable Keypoints; it looks for keypoints by comparing the brightness of nearby pixels, and storing it as binary values (1 if a pixel is brighter, 0 if not). also uses FAST to find corners
**Homography:** A 3x3 transformation matrix that maps an image onto another plane (basically the image viewed from a different perspective)
**Repeatability Rate:** checking how many keypoints from the original image were detected in the transformed image
- Note: we only check the original image's project points onto image N, not the other way around
**Localization Error:** checking how far the keypoints in the transformed image were compared to their location in the original image

## For the Future
Compare Against: ALGD-ORB
- Trying Adaptive Threshold and Local Gray Difference-ORB (ALGD-ORB) 
Compare Against: Personally Created Feature Detector

Measure More Metrics:
- Precision
- Keypoint Distribution
- Test With Actual Visual Odometry

## Limitations and Constraints
in progress

## Results
Check out the graphs!
(in progress)

# Tech Stack
in progress

# Sources
https://docs.opencv.org/4.13.0/da/df5/tutorial_py_sift_intro.html
https://docs.opencv.org/4.13.0/da/df5/tutorial_py_sift_intro.html
in progress