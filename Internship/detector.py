import cv2
import argparse
import numpy as np

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image on disk")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])

#definition of the color boundaries.

# Boundary from very light blue color to very dark blue color in order 
# to detect not only the main part of the water body but also the contours.

boundaries = [	
	([0, 0, 0], [140, 150, 100])
]	

boundary = ([0,0,0], [140,150, 100])


(lower, upper) = boundary

lower = np.array(lower, dtype = "uint8")
upper = np.array(upper, dtype = "uint8")

def find_mask(image):
    return cv2.inRange(image, lower, upper)

def find_contours(mask):
    (_, cnts, hierarchy) = cv2.findContours(
            mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print("Found %d black shapes" % (len(cnts)))
    return cnts

#draw contours on an image
# @param {list[[int, int]]} an array of [int, int] points to draw
# @param {image} the image to draw the points on
def show_contours(contours, image):
    cv2.drawContours(image, contours, -1, (0, 0, 255), 2)

    cv2.imshow("contours", image)

def get_main_contour(contours):
    copy = contours.copy()
    copy.sort(key=len, reverse=True)
    return copy[0]


mask = find_mask(image)

contours = find_contours(mask)
main_contour = get_main_contour(contours) 
show_contours([main_contour], image)
key = cv2.waitKey(0)