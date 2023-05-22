import math

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
	height, width = int(rect[1][0]), int(rect[1][1])

	# top left, top right, bottom right, bottom left
	src_pts = np.int0(cv2.boxPoints(rect)).astype("float32")
	dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

	# Transformation Matrix
	M = cv2.getPerspectiveTransform(src_pts, dst_pts)
	# Rotate view
	warped = cv2.warpPerspective(img, M, (width, height))
	return width, height, rect[2], warped


def get_center_points(corners):
	center_points = []
	for marker_corners in corners:
		(cX, cY), (_, _), _ = cv2.minAreaRect(marker_corners[0])
		center_points.append([cX, cY])

	return np.array(center_points, dtype=np.float32)
