from abc import abstractmethod, ABC

from ArucoCrop.ArucoCropUtils import rotate_img, get_center_points


class ArucoArea(ABC):

	@staticmethod
	def rotate_and_crop(image, rel_corners):
		center_points = get_center_points(rel_corners)
		return rotate_img(image, center_points)

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
