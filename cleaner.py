import cv2 as cv

def clean_image(img):
    clahe = cv.createCLAHE(clipLimit=40.0)
    final = clahe.apply(img)
    #final = cv.GaussianBlur(final, (87, 87), 0)
    print(final.dtype)
    return final