import numpy as np
import cv2 as cv



class Vision:

    # Properties
    template = None
    template_w = 0
    template_h = 0
    method = None

    # Constructor
    def __init__(self, template_path, method = cv.TM_CCOEFF_NORMED):
        # Load the template
        self.template = cv.imread(template_path)

        # Save the dimensions of template
        self.template_w = self.template.shape[1]
        self.template_h = self.template.shape[0]

        # Choose a method
        self.method = method

    def find(self, screenshot, threshold = 0.3, debug_mode = None):
        # Run the OpenCV algorithm
        result = cv.matchTemplate(screenshot, self.template, self.method)

        locations = np.where(result >= threshold)
        locations = list(zip(*locations[::-1]))

        rectangles = []
        for loc in locations:
            rect = [int(loc[0]),int(loc[1]),self.template_w, self.template_h]
            rectangles.append(rect)
            rectangles.append(rect)

        # group rectangles together so they do not overlap
        rectangles,weights = cv.groupRectangles(rectangles,1,0.5)
        print(rectangles)
        points = []
        if len(rectangles):
            line_color = (0,255,0)
            line_type = cv.LINE_4
            line_thickness = 2
            marker_color = (0,0,255)
            marker_type = cv.MARKER_CROSS
            # need to loop over all the locations and draw their rectangle
            for (x,y,w,h) in rectangles:
                # Determine the center position
                center_x = x + int(w / 2)
                center_y = y + int(h / 2)
                # Save the points
                points.append((center_x, center_y))
                if debug_mode == "rectangles":
                    # Determine the box positions
                    top_left = (x,y)
                    bottom_right = (x + w,y + h)
                    # Draw the box
                    cv.rectangle(screenshot,top_left,bottom_right,line_color,line_thickness,line_type)
                elif debug_mode == "points":

                    # Draw the point
                    cv.drawMarker(screenshot,(center_x,center_y),marker_color,marker_type)
        cv.imshow("Debug", screenshot)
        return points
