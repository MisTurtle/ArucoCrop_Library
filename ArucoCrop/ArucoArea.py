import math
from abc import abstractmethod, ABC
from collections import Counter

import cv2
import numpy as np

from ArucoCrop.ArucoCropUtils import rotate_img, get_center_points


class ArucoArea(ABC):

	@staticmethod
	def rotate_and_crop(image, rel_corners):
		center_points = get_center_points(rel_corners)
		straight = rotate_img(image, center_points)
		top_left_corner_ids = []
		for corners in rel_corners:
			corners = corners.reshape(-1, 2).astype(np.int0)
			minLen, topLeftId = float('inf'), 0
			for cid, corner in enumerate(corners):
				length = math.sqrt(corner[0] ** 2 + corner[1] ** 2)
				if length < minLen:
					minLen = length
					topLeftId = cid
			top_left_corner_ids.append(topLeftId)
		counter = Counter(top_left_corner_ids)
		common = counter.most_common(1)[0]
		match common:
			case 0:  # Top left for most markers is already their top left
				return straight
			case 1:  # Top left is mostly top right
				angle = cv2.ROTATE_90_COUNTERCLOCKWISE
			case 2:  # Top left is mostly bottom right
				angle = cv2.ROTATE_180
			case _:
				angle = cv2.ROTATE_90_CLOCKWISE
		return cv2.rotate(straight, angle)

	def __init__(self, name: str, aruco_id: int, marker_count: int):
		self.name = name
		self.aruco_id = aruco_id
		self.marker_count = marker_count
		self.offlineTicks = 0

	def get_name(self) -> str:
		return self.name

	def get_aruco_id(self) -> int:
		return self.aruco_id

	def get_offline_ticks(self) -> int:
		return self.offlineTicks

	def is_visible(self, found_ids: list[int]) -> bool:
		return len(list(filter(lambda x: x == self.aruco_id, found_ids))) == self.marker_count

	def filter(self, founds_ids: list[int], corners):
		if founds_ids is None:
			return []
		rel_corners = []
		for i in range(len(founds_ids)):
			if founds_ids[i] == self.aruco_id:
				rel_corners.append(corners[i])
		return rel_corners

	@abstractmethod
	def process(self, image, rel_corners):
		pass


class CallbackArucoArea(ArucoArea):

	def __init__(self, name: str, aruco_id: int, marker_count: int, process_method):
		print(process_method)
		super().__init__(name, aruco_id, marker_count)
		self.pm = process_method

	def process(self, image, rel_corners):
		return self.pm(self, image, rel_corners)
