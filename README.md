# Credits
- credit to https://github.com/hpatches/hpatches-dataset/tree/master for sequence images

# Issues
- say u have a large object (mechanical keyboard), it will detect the distinct keys rather than treat the keyboard as a whole; raising contrast does not help
- too high a gaussian blur makes it not detect anything at all 
- not even related to detectors but y did opencv remove brisk from their most recent version and put it in a separate library no one was complaining that it was in open cv v4 so whyd they get rid of it why why why

# Iteration 1: Desperation bro
- gaussian blur SUCKED changing to a different rule for orb SUCKED RHSFHFSDLJKSDFKJLSDFJLKFSDJLK

# Testing Different Feature Detectors
- SIFT(Scale-Invariant Feature Transform)
- ORB (Oriented FAST and Rotated BRIEF)
- BRISK

Track repeatibility rate (can it consistently get a point), precision(true pos vs false pos, etc.) and recall (how much of the real points did it actually get), localization error (dist between keypoint and actual point irl), keypoints per second

# Key Definitions
SIFT:
ORB:
BRISK:
Homography:
Keypoint:
Descriptor:
Repeatibility Rate: 
- Note: we only check the original image's project points onto image N, not the other way around
Localization Error:

# For the Future
Iteration 2: ALGD-ORB
- Trying Adaptive Threshold and Local Gray Difference-ORB (ALGD-ORB) 

