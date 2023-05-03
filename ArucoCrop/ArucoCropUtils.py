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


def debug(what: str):
	print('> ArucoCrop Debug : {}'.format(what))


def rotate_img(img, points):
	"""
	:param img: Image to rotate
	:param points: Points to turn into a straight rectangle
	:return: Img Width, Img Height, Read Angle, Rotated Image
	"""
	rect = cv2.minAreaRect(points)
	width, height = int(rect[1][0]), int(rect[1][1])

	# bottom left, top left, top right, bottom right
	src_pts = np.int0(cv2.boxPoints(rect)).astype("float32")
	dst_pts = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype="float32")

	# Transformation Matrix
	M = cv2.getPerspectiveTransform(src_pts, dst_pts)
	# Rotate view
	warped = cv2.warpPerspective(img, M, (width, height))

	return width, height, rect[2], warped


def get_center_points(corners) -> np.ndarray:
	center_points = []
	for markerCorners in corners:
		(tL, _, bR, _) = markerCorners.reshape((4, 2))
		tL = int(tL[0]), int(tL[1])
		bR = int(bR[0]), int(bR[1])
		center_points.append([[(bR[0] + tL[0]) / 2, (bR[1] + tL[1]) / 2]])
	return np.array(center_points, dtype="float32")


