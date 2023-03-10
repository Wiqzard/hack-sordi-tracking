from typing import List, Type

import numpy as np

from detection.detection_tools import Detections
from onemetric.cv.utils.iou import box_iou_batch

# from bytetrack.yolox.tracker.byte_tracker import BYTETracker, STrack


# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((detections.xyxy, detections.confidence[:, np.newaxis]))


# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
# def tracks2boxes(tracks: List[Type[STrack]]) -> np.ndarray:
#    return np.array([
#        track.tlbr
#        for track
#        in tracks
#    ], dtype=float)
#
def tracks2boxes(tracks) -> np.ndarray:
    """
    It takes a list of `Track` objects and returns a numpy array of their bounding boxes

    :param tracks: list of Track objects
    :return: The top left bottom right coordinates of the bounding box
    """
    return np.array([track.tlbr for track in tracks], dtype=float)


# matches our bounding boxes with predictions
def match_detections_with_tracks(detections: Detections, tracks) -> Detections:
    detection_boxes = detections.xyxy
    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detection_boxes)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id

    return tracker_ids


# def match_detections_with_tracks(
#    detections: Detections,
#    tracks: List[STrack]
# ) -> Detections:
#    detection_boxes = detections.xyxy
#    tracks_boxes = tracks2boxes(tracks=tracks)
#    iou = box_iou_batch(tracks_boxes, detection_boxes)
#    track2detection = np.argmax(iou, axis=1)
#
#    tracker_ids = [None] * len(detections)
#
#    for tracker_index, detection_index in enumerate(track2detection):
#        if iou[tracker_index, detection_index] != 0:
#            tracker_ids[detection_index] = tracks[tracker_index].track_id
#
#    return tracker_ids
