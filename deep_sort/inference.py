from collections import namedtuple
from typing import List

from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.generate_detections import create_box_encoder

from dataclasses import dataclass
import numpy as np

from deep_sort.utils import xyxy2xywhn

# bbox - ((x_min, y_min, w, h), score, det_id)
ObjDetection = namedtuple('ObjDetection', 'bbox score det_id')


@dataclass
class TrackedBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    score: float
    tracking_id: int
    det_id: str
    xywhn: List = ()

    def __post_init__(self):
        self.width = self.x_max - self.x_min
        self.height = self.y_max - self.y_min
        self.bbox = (self.x_min, self.y_min, self.width, self.height)


class DeepsortTracker:

    def __init__(self, model_path: str, max_cosine_distance=0.7, nn_budget=None):
        self.encoder = create_box_encoder(model_path, batch_size=1)
        self.metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric=self.metric, max_iou_distance=0.6, max_age=3, n_init=3)

    def track_boxes(self, frame, boxes: List[ObjDetection]):
        """

        bbox - ((x_min, y_min, w, h), score, det_id)
        """
        features = np.array(self.encoder(frame, [box.bbox for box in boxes]))
        detections = [
            Detection(bbox.bbox, bbox.score, feature, bbox.det_id)
            for bbox, feature
            in zip(boxes, features)
        ]

        # Pass detections to the deepsort object and obtain the track information.
        self.tracker.predict()
        self.tracker.update(detections)

        # Obtain info from the tracks
        tracked_bboxes = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.detection.to_tlbr()  # Get the corrected/predicted bounding box
            tracking_id = track.track_id  # Get the ID for the particular track

            x_min, y_min, x_max, y_max = [min(max(int(i), 0), max(frame.shape)) for i in bbox.tolist()]

            det_obj = TrackedBox(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                score=track.detection.confidence,
                tracking_id=tracking_id,
                det_id=track.detection.det_id,
                xywhn=xyxy2xywhn(np.array([[y_min, y_min, x_max, y_max]], dtype=float), h=frame.shape[0], w=frame.shape[1]).tolist()[0]
            )
            tracked_bboxes.append(det_obj)  # Structure data, that we could use it with our draw_bbox function
        return tracked_bboxes

    def reset(self):
        self.tracker = Tracker(self.metric)