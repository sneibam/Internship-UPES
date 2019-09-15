# Importation of necessary packages

import numpy as np
import argparse
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", help="type of detection you want to make | either water or green")
ap.add_argument("-i", "--image", help = "path to the image on disk")
ap.add_argument("-c", "--color", help="color ranges for the water body, either blue focused or green focused")
args = vars(ap.parse_args())

typeDetection = args["type"]
imageFileName = args["image"]
typeColors = args["color"]

# load the image
image = cv2.imread(args["image"])

#definition of the color boundaries.

# Boundary from very light blue color to very dark blue color in order 
# to detect not only the main part of the water body but also the contours.

if typeDetection == "water":

	if typeColors == 'b':
		boundary = ([230,150,30], [255,255, 240])
		#boundary = ([200,0,0], [255,255, 240])
	elif typeColors == 'g':
		boundary = ([180,180,70], [255,255, 115])
	elif typeColors == 'db1':
		boundary = ([100,55,33], [168,155, 62])
	elif typeColors == 'db2':
		boundary = ([130,79,39], [190,159, 70])
else:
	cv2.imshow("images", image)
	cv2.waitKey(0)
	boundary = ([0, 50, 0], [120, 150, 107])


(lower, upper) = boundary
	
# create Numpy arrays for the boundaries

lower = np.array(lower, dtype = "uint8")
upper = np.array(upper, dtype = "uint8")

# find the colors within the specified boundaries and apply the mask

mask = cv2.inRange(image, lower, upper)
output = cv2.bitwise_and(image, image, mask = mask)

# show the images

cv2.imshow("images", np.hstack([image, output]))
cv2.waitKey(0)

imgray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
cv2.imshow("image gray", imgray)
cv2.waitKey(0)

ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, _ = cv2.findContours(imgray.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Drawing the countours

cv2.drawContours(image, contours, -1, (0,0, 255), 3)

# Computation of the average length of contours
 
avg = 0.0
for c in contours:
	avg += len(c)

avg /= len(contours)


#Initialize the values that will help us keep track of the largest contours
largestContourSize = -1
largestContourIndex = -1

print('countour : ' + str(avg))

for i, c in enumerate(contours):
	# Keep track of the largest contours
	if len(c) > largestContourSize:
		largestContourSize = len(c)
		largestContourIndex = i


x, y, w, h = cv2.boundingRect(contours[largestContourIndex])

# Moments

moments = cv2.moments(contours[largestContourIndex])
print(moments)

## Computation of centroids and area

cv2.imshow("Image with contours", image)
cv2.waitKey(0)

cx = int(moments['m10']/moments['m00'])
cy = int(moments['m01']/moments['m00'])

print("(" + str(cx) + "," + str(cy) + ")")

print("Area : " + str(moments['m00']))

area = moments['m00']

# print area of the main contour

print(str(largestContourSize) + " " + str(largestContourIndex))

print('%0.2f cm wide x %0.2f cm tall' % (w, h))


#Writing the results into an existing file

f = open("data.csv", "a")

f.write("\n" + imageFileName + "," + imageFileName.split('/')[1][6:10] + "," + str(largestContourSize) + "," + str(avg) + "," + str(w) + "," + str(h) + "," 
		+ str(cx) + "," + str(cy) + "," + str(area))

f.close()


