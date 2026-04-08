"""YOLO configuration for v11 planner scripts."""
import pathlib

# Model weights bundled inside v11/
MODEL_PATH = str(pathlib.Path(__file__).parent / "yolo_best.pt")

# 13 BFMC classes
CLASS_NAMES = [
    'one_way_road_sign',
    'highway_entrance_sign',
    'stop_sign',
    'roundabout_sign',
    'parking_sign',
    'crosswalk_sign',
    'no_entry_road_sign',
    'highway_exit_sign',
    'priority_sign',
    'traffic_light',
    'highway_separator',
    'pedestrian',
    'car',
]

CONFIDENCE_THRESHOLD = 0.60
IOU_THRESHOLD        = 0.45
