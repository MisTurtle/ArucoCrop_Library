import math
from scipy.spatial import distance as dist
import cv2
import numpy as np


def throw_error(err: str, **kwargs):
	err = '> ArucoCrop Error : {}'.format(err)
	if kwargs.get('fatal', False):
		raise Exception(err)
	else:
		print(err, end=kwargs.get('end', '\n'))
		if kwargs.get('interact', False):
			input('Press enter to continue...')


def debug(what: str, prefix: str = "> ArucoCrop Debug : "):
	print('{} {}'.format(prefix, what))


def rotate_img(img, points):
	"""
	:param img: Image to rotate
	:param points: Points to turn into a straight rectangle
	:return: Img Width, Img Height, Read Angle, Rotated Image
	"""
	rect = cv2.minAreaRect(points)
	width, height = int(rect[1][0]), int(rect[1][1])

	# top left, top right, bottom right, bottom left
	src_pts = points.astype("float32")
	dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

	# Transformation Matrix
	M = cv2.getPerspectiveTransform(src_pts, dst_pts)
	# Rotate view
	warped = cv2.warpPerspective(img, M, (width, height))
	return width, height, rect[2], warped


def order_points(pts):
	"""
	Source : https://pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
	:param pts:
	:return:
	"""
	# sort the points based on their x-coordinates
	xSorted = pts[np.argsort(pts[:, 0]), :]
	# grab the left-most and right-most points from the sorted
	# x-roodinate points
	leftMost = xSorted[:2, :]
	rightMost = xSorted[2:, :]
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
	leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
	(tl, bl) = leftMost
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
	D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
	(br, tr) = rightMost[np.argsort(D)[::-1], :]
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order
	return np.array([tl, tr, br, bl], dtype="float32")


def get_center_points(corners):
	centers = []
	for corner in corners:
		(cX, cY), (_, _), _ = cv2.minAreaRect(corner[0])
		centers.append([cX, cY])
	return order_points(np.array(centers).astype(int)).astype(int)
# center_points = []
	# for marker_corners in corners:
	# 	(cX, cY), (_, _), _ = cv2.minAreaRect(marker_corners[0])
	# 	center_points.append([cX, cY])
	#
	# return np.array(center_points, dtype=np.float32)
