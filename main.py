import numpy as np
import cv2 as cv
from windows_capture import WindowCapture
from vision import Vision


wincap = WindowCapture()
vision_cookie = Vision("Images/Cookie Clicker/cookie.png",method = cv.TM_CCOEFF_NORMED)

# region Functions
# get grayscale image


def get_grayscale(image):
    return cv.cvtColor(image,cv.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv.medianBlur(image,5)


# thresholding
def thresholding(image):
    return cv.threshold(image,0,255,cv.THRESH_BINARY + cv.THRESH_OTSU)[1]


# dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv.dilate(image,kernel,iterations=1)


# erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv.erode(image,kernel,iterations=1)


# opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv.morphologyEx(image,cv.MORPH_OPEN,kernel)


# canny edge detection
def canny(image):
    return cv.Canny(image,100,200)

# endregion


# Main Loop
while True:
    # Get an updated image of the game
    screenshot = wincap.get_screenshot()

    # Find the object position
    points = vision_cookie.find(screenshot, threshold = 0.6, debug_mode = "rectangles")

    # press q with the output window focused to exit
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord("q"):
        cv.destroyAllWindows()
        break

print("Done")
