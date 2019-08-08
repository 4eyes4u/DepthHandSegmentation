import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
from scipy.ndimage.measurements import label

DEPTH = 1  # flag for debugging
COLOR = 2  # flag for debugging

# couldn't make a dictionary of lists :(
MAPPING_RED = [255, 0, 0]
MAPPING_GREEN = [0, 255, 0]
MAPPING_BLUE = [0, 0, 255]
MAPPING_YELLOW = [255, 255, 0]
MAPPING_VIOLET = [255, 0, 255]
MAPPING_GOLD = [255, 215, 0]
MAPPING_ORANGE = [255, 128, 0]


def display_image(img, name=None):
    cv2.imshow("test_image" if name is None else name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_depth(walk_dir):
    """
    Visualizing depth images that are in 16 bits.
    """

    for root, dirs, files in os.walk(walk_dir):
        for f in files:
            if "png" in f:
                img = Image.open(os.path.join(root, f))
                pixels = np.array(img)
                plt.imshow(pixels, cmap='gray', vmin=0, vmax=max(max(row) for row in pixels))
                plt.show()


def display_color(path):
    """
    Hand segmentation.
    """
    def keep_largest_component(color):
        """
        Finding largest component of pixels.

        Returns:
            -coordinates in shape (2, N).
        """

        color[color != [0, 0, 0]] = 255  # color now has [0, 0, 0] or [255, 255, 255]
        color = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        labeled, num_components = label(color, structure=np.ones((3, 3), dtype=np.int32))

        pixels = [None] * num_components
        for i in range(num_components):
            rows, cols = np.where(labeled == i + 1)
            pixels[i] = np.vstack((rows, cols)).T
        lens = np.array([len(p) for p in pixels])

        return pixels[np.argsort(lens)[-1]].T

    def extract_color2(img_hsv, first_range, second_range):
        """
        For colors on bounds in HSV.

        Returns:
            -masked image.
        """

        first_range = np.array(first_range)
        second_range = np.array(second_range)
        mask1 = cv2.inRange(img_hsv, *first_range)
        mask2 = cv2.inRange(img_hsv, *second_range)
        mask = cv2.bitwise_or(mask1, mask2)

        ret = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)

        return ret

    def extract_color(img_hsv, lower, upper):
        """
        For colors not on bounds in HSV.

        Returns:
            -masked image.
        """

        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(img_hsv, lower, upper)
        ret = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)

        return ret

    img_frame = Image.open(path)
    img_rgb = np.array(img_frame)
    img_rgb = cv2.GaussianBlur(img_rgb, (7, 7), 0)  # averaging noise

    # different color schemes
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_hsv[:, :, 2] = 255  # maximal value -- fixing brightness
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    placeholder_rgb = np.zeros_like(img_rgb)  # will hold segmentation

    # extracting hand from background
    _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    img_bgr[thresh == 0] = 0
    img_hsv[thresh == 0] = 0

    print(img_hsv[232, 810])
    print(img_hsv[228, 822])
    print(img_hsv[262, 804])
    print(img_hsv[257, 776])

    # segmentation
    color_red = extract_color(img_hsv, lower=[170, 50, 255], upper=[178, 255, 255])  # RED
    pixels_red = keep_largest_component(color_red)
    placeholder_rgb[pixels_red[0], pixels_red[1]] = MAPPING_RED

    color_green = extract_color(img_hsv, lower=[33, 130, 255], upper=[60, 200, 255])  # GREEN
    pixels_green = keep_largest_component(color_green)
    placeholder_rgb[pixels_green[0], pixels_green[1]] = MAPPING_GREEN

    color_violet = extract_color(img_hsv, lower=[120, 20, 255], upper=[165, 200, 255])  # VIOLET
    pixels_violet = keep_largest_component(color_violet)
    placeholder_rgb[pixels_violet[0], pixels_violet[1]] = MAPPING_VIOLET

    color_blue = extract_color(img_hsv, lower=[100, 90, 255], upper=[120, 255, 255])  # BLUE
    pixels_blue = keep_largest_component(color_blue)
    placeholder_rgb[pixels_blue[0], pixels_blue[1]] = MAPPING_BLUE

    color_yellow = extract_color(img_hsv, lower=[5, 200, 255], upper=[20, 255, 255])  # YELLOW
    rows_yellow, cols_yellow, _ = np.where(color_yellow != [0, 0, 0])
    placeholder_rgb[rows_yellow, cols_yellow] = MAPPING_YELLOW

    color_orange = extract_color2(img_hsv,
                                  first_range=[[178, 150, 255], [179, 255, 255]],
                                  second_range=[[0, 150, 255], [3, 255, 255]])
    pixels_orange = keep_largest_component(color_orange)
    placeholder_rgb[pixels_orange[0], pixels_orange[1]] = MAPPING_ORANGE
    # color_gold = extract_color(img_hsv, lower=[0, 0, 0], upper=[0, 0, 0])  # TODO: find ranges

    display_image(cv2.cvtColor(placeholder_rgb, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    FLAGS = 2
    if FLAGS & DEPTH:
        display_depth("C:\\Program Files\\Azure Kinect SDK v1.1.1\\tools\\depth")

    if FLAGS & COLOR:
        color_walk = r"C:\Program Files\Azure Kinect SDK v1.1.1\tools\rgb"
        for root, dirs, files in os.walk(color_walk):
            for f in files:
                if "png" in f:
                    display_color(os.path.join(root, f))

    display_color(r"C:\Program Files\Azure Kinect SDK v1.1.1\tools\rgb\rgb0001.png")
