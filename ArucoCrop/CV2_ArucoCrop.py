from typing import Any, Union

from collections import OrderedDict

from cv2 import aruco

from ArucoCrop.ArucoCropUtils import *
from ArucoCrop.ArucoArea import *


arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
arucoParams = cv2.aruco.DetectorParameters_create()
arucoAreas = OrderedDict()
_debug = False
_debugPrefix = ""


def init(**kwargs):
	_dict = kwargs.get('dict', None)
	_params = kwargs.get('parameters', None)
	_areas = kwargs.get('areas', None)
	_debugState = kwargs.get('debug', False)
	_prefix = kwargs.get('debug_prefix', '> ArucoCrop Debug : ')

	set_DebugState(_debugState)
	set_DebugPrefix(_prefix)

	if isinstance(_dict, cv2.aruco_Dictionary):
		set_ArucoDictionary(_dict)
	elif _dict is not None:
		throw_error('`dict` must be of type cv2.aruco_Dictionary', fatal=True)

	if isinstance(_params, cv2.aruco_DetectorParameters):
		set_ArucoDetectorParameters(_params)
	elif _params is not None:
		throw_error('`parameters` must be of type cv2.aruco_DetectorParameters', fatal=True)

	if isinstance(_areas, list):
		set_ArucoAreas(_areas)
	elif _areas is not None:
		throw_error('`areas` must be of type list')

	if _debugState:
		print("--- --- END : INIT --- ---")


def set_DebugState(_state: bool):
	global _debug
	_debug = _state


def set_DebugPrefix(_prefix: str):
	global _debugPrefix
	_debugPrefix = _prefix


def set_ArucoDictionary(_dict: cv2.aruco_Dictionary):
	global arucoDict, _debug
	arucoDict = _dict
	if _debug:
		debug('Successfully set Aruco Dictionary', _debugPrefix)


def set_ArucoDetectorParameters(_params: cv2.aruco_DetectorParameters):
	global arucoParams, _debug
	arucoParams = _params
	if _debug:
		debug('Successfully set Aruco Detector Parameters', _debugPrefix)


def set_ArucoAreas(_areas: list):
	clear_ArucoAreas()
	register_ArucoAreas(_areas)


def register_ArucoAreas(_areas: list, _override: bool = False):
	global _debug
	for _area in _areas:
		if isinstance(_area, ArucoArea):
			register_ArucoArea(_area, _override)
		else:
			throw_error('Trying to register ArucoArea of type {}'.format(type(_area)), fatal=True)
	if _debug:
		debug("Successfully Registered {} Aruco Areas".format(len(_areas)), _debugPrefix)


def register_ArucoArea(_area: ArucoArea, _override: bool = False):
	global _debug
	old = get_ArucoArea(_area.get_name())
	if not _override and old is not None:
		throw_error("Trying to override area named `{}`".format(old.get_name()), fatal=True)
	if old is not None:
		unregister_ArucoArea(old)
	global arucoAreas
	arucoAreas[_area.get_name()] = _area
	if _debug:
		debug("Successfully Registered Area {}".format(_area.get_name()), _debugPrefix)


def get_ArucoArea(_area_name: str, _default: Any = None) -> Union[ArucoArea, Any]:
	if _area_name == "":
		throw_error("No valid name passed when retrieving an area")
		return _default
	global arucoAreas, _debug
	if _debug:
		area = arucoAreas.get(_area_name, _default)
		debug("Return value after getting area named `{}` : {}".format(_area_name, area), _debugPrefix)
		return area
	return arucoAreas.get(_area_name, _default)


def clear_ArucoAreas():
	global arucoAreas, _debug
	for _area in arucoAreas.values():
		unregister_ArucoArea(_area)
	if _debug:
		debug("Cleared All Aruco Areas", _debugPrefix)


def unregister_ArucoArea(_area: Union[ArucoArea, str]):
	name = _area
	if isinstance(name, ArucoArea):
		name = _area.get_name()
	elif not isinstance(name, str):
		throw_error("No area or area name given to be unregistered")
		return
	global arucoAreas, _debug

	if _debug:
		if arucoAreas.get(name, None) is not None:
			debug("Successfully Unregistered Area `{}`".format(name), _debugPrefix)
		else:
			debug("No area named {} could be found".format(name), _debugPrefix)
	del arucoAreas[name]


def process_frame(_frame: np.ndarray) -> list:
	global _debug, _debugPrefix, arucoDict, arucoAreas, arucoParams
	_processed = []
	(_corners, _ids, _rejected) = cv2.aruco.detectMarkers(_frame, arucoDict, parameters=arucoParams)
	if _ids is None:
		if _debug:
			debug("No marker was detected", _debugPrefix)
		return _processed
	else:
		debug("Detected markers: {}".format(" ; ".join(_ids)), _debugPrefix)
	area: ArucoArea
	for area in arucoAreas.values():
		if not area.is_visible(_ids):
			if _debug:
				debug("Area `{}` is invisible".format(area.get_name()), _debugPrefix)
			continue
		rel_corners = area.filter(_ids, _corners)
		result = area.process(_frame, rel_corners)
		_processed.append(area.get_aruco_id())
		if _debug:
			debug("Successfully Processed Area `{}`".format(area.get_name()), _debugPrefix)
			#cv2.imshow(area.get_name(), result)
			cv2.waitKey(0)
	return _processed
