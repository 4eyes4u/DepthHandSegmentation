import argparse
import cv2
import numpy as np
import os
import sys

from PIL import Image
from matplotlib import pyplot as plt
from scipy.ndimage.measurements import label
from scipy.spatial import ConvexHull

DEPTH = 1  # flag for debugging
COLOR = 2  # flag for debugging

# couldn't make a dictionary of lists :(
MAPPING_RED = [255, 0, 0]
MAPPING_GREEN = [0, 255, 0]
MAPPING_BLUE = [0, 0, 255]
MAPPING_YELLOW = [255, 255, 0]
MAPPING_VIOLET = [255, 0, 255]
MAPPING_WHITE = [255, 255, 255]
MAPPING_ORANGE = [255, 128, 0]


def center_crop(img, cropx, cropy):
    y, x, _ = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty: starty + cropy, startx: startx + cropx, ...]


def print_pixel(img, x, y):
    print(img[y, x])


def gamma_correction(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


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
                plt.imshow(pixels, cmap='jet', vmin=0, vmax=max(max(row) for row in pixels))
                plt.show()


def display_color(path):
    """
    Hand segmentation.
    """

    def fill_all_components(color, img, mapping):
        color[color != [0, 0, 0]] = 255
        color = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        labeled, num_components = label(color, structure=np.ones((3, 3), dtype=np.int32))

        for i in range(num_components):
            rows, cols = np.where(labeled == i + 1)
            pixels = np.vstack((rows, cols))
            img = fill_convex_hull(pixels, img, mapping)

        return img

    def keep_largest_component(color, keep=1, sample=False):
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

        indices = np.argsort(lens)[::-1]
        if keep != -1:
            indices = [indices[keep - 1]] if sample else indices[:keep]

        taken = []
        for i in indices:
            taken.append(pixels[i])

        return np.vstack(taken).T

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

    def extract_gold(img_hsv, img_rgb, lower, upper):
        """
        For colors not on bounds in HSV.

        Returns:
            -masked image.
        """

        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(img_hsv, lower, upper)
        ret = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)
        img_hsv[img_rgb[:, :, 2] < 15] = 0

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

    def fill_convex_hull(pixels, img, mapping):
        assert pixels.shape[0] == 2
        tmp = pixels[0].copy()
        pixels[0] = pixels[1]
        pixels[1] = tmp
        pixels = pixels.T

        ch = ConvexHull(pixels)
        pixels_idx = np.asarray(ch.vertices, dtype=np.int32)
        pts = np.asarray([pixels[i] for i in pixels_idx])
        pts = np.int32([pts])
        img = cv2.fillPoly(img, pts, mapping)

        return img

    # fetching data
    img_frame = Image.open(path)
    img_rgb = np.array(img_frame)
    # img_rgb = center_crop(img_rgb, 640, 360)
    # img_rgb = gamma_correction(img_rgb, 1.25)
    img_rgb = cv2.GaussianBlur(img_rgb, (7, 7), 0)  # averaging noise

    # different color schemes
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_hsv[:, :, 2] = 255  # maximal value -- fixing brightness
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    placeholder_rgb = np.zeros_like(img_rgb)  # will hold segmentation
    # cv2.imwrite('valued.png', img_bgr)

    # extracting hand from background
    _, thresh = cv2.threshold(img_gray, args.threshold, 255, cv2.THRESH_BINARY_INV)
    img_bgr[thresh == 0] = 0
    img_hsv[thresh == 0] = 0

    # segmentation
    color_red = extract_color(img_hsv, lower=[170, 50, 0], upper=[178, 255, 255])  # RED
    pixels_red = keep_largest_component(color_red, keep=args.red_components, sample=False)
    placeholder_rgb[pixels_red[0], pixels_red[1]] = MAPPING_RED
    centroid_red = np.mean(pixels_red, axis=1).astype(np.int32)
    placeholder_rgb = fill_convex_hull(pixels_red, placeholder_rgb, MAPPING_RED)

    color_green = extract_color(img_hsv, lower=[20, 100, 0], upper=[70, 200, 255])  # GREEN
    pixels_green = keep_largest_component(color_green)
    placeholder_rgb[pixels_green[0], pixels_green[1]] = MAPPING_GREEN
    centroid_green = np.mean(pixels_green, axis=1).astype(np.int32)
    placeholder_rgb = fill_convex_hull(pixels_green, placeholder_rgb, MAPPING_GREEN)

    color_violet = extract_color(img_hsv, lower=[120, 20, 0], upper=[165, 200, 255])  # VIOLET
    pixels_violet = keep_largest_component(color_violet)
    placeholder_rgb[pixels_violet[0], pixels_violet[1]] = MAPPING_VIOLET
    centroid_violet = np.mean(pixels_violet, axis=1).astype(np.int32)
    placeholder_rgb = fill_convex_hull(pixels_violet, placeholder_rgb, MAPPING_VIOLET)

    color_blue = extract_color(img_hsv, lower=[100, 90, 0], upper=[120, 255, 255])  # BLUE
    pixels_blue = keep_largest_component(color_blue)
    placeholder_rgb[pixels_blue[0], pixels_blue[1]] = MAPPING_BLUE
    centroid_blue = np.mean(pixels_blue, axis=1).astype(np.int32)
    placeholder_rgb = fill_convex_hull(pixels_blue, placeholder_rgb, MAPPING_BLUE)

    color_yellow = extract_color(img_hsv, lower=[5, 180, 0], upper=[20, 255, 255])  # YELLOW
    rows_yellow, cols_yellow, _ = np.where(color_yellow != [0, 0, 0])
    placeholder_rgb[rows_yellow, cols_yellow] = MAPPING_YELLOW

    color_orange = extract_color(img_hsv, lower=[0, 150, 0], upper=[3, 200, 255])
    # color_orange = extract_color2(img_hsv,
    #                               first_range=[[178, 150, 0], [179, 200, 255]],
    #                               second_range=[[0, 150, 0], [3, 200, 255]])
    pixels_orange = keep_largest_component(color_orange)
    placeholder_rgb[pixels_orange[0], pixels_orange[1]] = MAPPING_ORANGE
    centroid_orange = np.mean(pixels_orange, axis=1).astype(np.int32)
    placeholder_rgb = fill_convex_hull(pixels_orange, placeholder_rgb, MAPPING_ORANGE)

    # drawing centroids for every class
    cv2.circle(img_rgb, (centroid_red[1], centroid_red[0]), 15, (255, 255, 255))
    cv2.circle(img_rgb, (centroid_violet[1], centroid_violet[0]), 15, (255, 255, 255))
    cv2.circle(img_rgb, (centroid_green[1], centroid_green[0]), 15, (255, 255, 255))
    cv2.circle(img_rgb, (centroid_blue[1], centroid_blue[0]), 15, (255, 255, 255))
    cv2.circle(img_rgb, (centroid_orange[1], centroid_orange[0]), 15, (255, 255, 255))

    # cv2.circle(img_rgb, (centroid_orange[1], centroid_orange[0]), 15, (255, 255, 255))  # centroids on every class
    # display_image(cv2.cvtColor(placeholder_rgb, cv2.COLOR_RGB2BGR), path.split(os.sep)[-1])

    path_split, file_name = os.path.split(path)
    file_name = file_name.replace("c", "s")
    path = os.path.join(path_split, file_name)
    cv2.imwrite(path, cv2.cvtColor(placeholder_rgb, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--walk_dir", help="Directory with images", type=str, required=True)
    # parser.add_argument("--red_components", help="1 or 2 depending on finger overlap", type=int, required=True)
    # parser.add_argument('--threshold', help='Threshold for binarizing image (background extraction)', type=int, required=True)
    # args = parser.parse_args()

    FLAGS = 1
    if FLAGS & DEPTH:
        depth_walk = r"C:\Program Files\Azure Kinect SDK v1.1.1\tools\depth"
        display_depth(depth_walk)

    # if FLAGS & COLOR:
    #     for root, dirs, files in os.walk(args.walk_dir):
    #         for f in files:
    #             if "png" in f:
    #                 try:
    #                     path = os.path.join(root, f)
    #                     display_color(path)
    #                     print("{} finished".format(path))
    #                 except:
    #                     print("{} failed".format(path))
    #                     continue
