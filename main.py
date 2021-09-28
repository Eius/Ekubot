import cv2 as cv
from windows_capture import WindowCapture
import numpy as np

wincap = WindowCapture()
img = cv.imread('Images/Albion_Debug.png')
template = cv.imread("Images/Albion_Debug_Template.png")
match_threshold = 0.2

match_w = template.shape[1]
match_h = template.shape[0]


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


# template matching
def match_template(image,template):
    return cv.matchTemplate(image,template,cv.TM_SQDIFF_NORMED)
# endregion

# region Processed images


gray = get_grayscale(img)
thresh = thresholding(gray)
opening = opening(gray)
canny = canny(gray)
# endregion

# region Template matching
result = match_template(img,template)
locations = np.where(result <= match_threshold)
locations = list(zip(*locations[::-1]))

# first we need to create the list of [x, y, w, h] rectangles
rectangles = []
for loc in locations:
    rect = [int(loc[0]), int(loc[1]), match_w, match_h]
    rectangles.append(rect)
    rectangles.append(rect)

rectangles, weights = cv.groupRectangles(rectangles, 1, 0.1)
if len(rectangles):
    line_color = (0, 255, 0)
    line_type = cv.LINE_4
    line_thickness = 2

    # need to loop over all the locations and draw their rectangle
    for (x, y, w, h) in rectangles:
        # Determine the box positions
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        # Draw the box
        cv.rectangle(img, top_left, bottom_right, line_color, line_thickness, line_type)

# endregion
cv.imshow("Debug", img)
cv.waitKey(0)
exit()
# Main Loop
while True:
    screenshot = wincap.get_screenshot()
    cv.imshow("Computer Vision", screenshot)

    # press q with the output window focused to exit
    # waits 1 ms every loop to process key presses
    if cv.waitKey(1) == ord("q"):
        cv.destroyAllWindows()
        break

print("Done")
