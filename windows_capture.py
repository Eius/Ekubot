import numpy as np
import win32con
import win32gui
import win32ui


class WindowCapture:

    # class properties
    w = 1920
    h = 1080
    hwnd = None
    cropped_x = 0
    cropped_y = 0
    offset_x = 0
    offset_y = 0

    # constructor
    def __init__(self, window_name = None):
        # find the handle for the window we want to capture
        # if no window name is given, capture the entire screen
        if window_name is None:
            self.hwnd = win32gui.GetDesktopWindow()
        else:
            #TODO: nedokáže nájsť Albion okno, problém je vysvetlený tu:
            # https://stackoverflow.com/questions/59014894/screenshots-taken-with-pywin32-some-times-get-a-black-imgaes-i-think-that-handl
            self.hwnd = win32gui.FindWindow(None,window_name)
            if not self.hwnd:
                raise Exception("Window not found: {}".format(window_name))

            # get the window size
            window_rect = win32gui.GetWindowRect(self.hwnd)
            self.w = window_rect[2] - window_rect[0]
            self.h = window_rect[3] - window_rect[1]

            # account for the window border and titlebar and cut them off
            border_pixels = 8
            titlebar_pixels = 30
            self.w = self.w - (border_pixels * 2)
            self.h = self.h - titlebar_pixels - border_pixels
            self.cropped_x = border_pixels
            self.cropped_y = titlebar_pixels

            # set the cropped coordinates offset so we can translate screenshot images into actual screen positions
            self.offset_x = window_rect[0] + self.cropped_x
            self.offset_y = window_rect[1] + self.cropped_y

    def get_screenshot(self):

        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj,self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0,0),(self.w, self.h),dcObj,(self.cropped_x, self.cropped_y),win32con.SRCCOPY)

        # dataBitMap.SaveBitmapFile(cDC, "debug.bmp")
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray,dtype="uint8")
        img.shape = (self.h, self.w, 4)

        # Free Resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd,wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        # Drop the alpha channel, or the cv.matchTemplate() will throw errors at you
        img = img[...,:3]
        # Make image C_CONTIGUOUS to avoid more errors
        img = np.ascontiguousarray(img)

        return img

    @staticmethod
    def list_window_names():
        def winEnumHandler(hwnd,ctx):
            if win32gui.IsWindowVisible(hwnd):
                print(hex(hwnd),win32gui.GetWindowText(hwnd))

        win32gui.EnumWindows(winEnumHandler,None)

    # translate a pixel position on a screenshot image to a pixel position on the screen
    # pos = (x, y)
    # WARNING_ if you move the window being captured after execution is started, this will
    # return incorrect coordinates, because the window position is only calculated in
    # the __init__ constructor

    # TODO: táto funkcia sa nikde nepoužíva
    def get_screen_position(self, pos):
        return (pos[0] + self.offset_x, pos[1] + self.offset_y)