import copy
import shutil

from sort.tracker import Sort
from deep_sort.inference import ObjDetection, DeepsortTracker
import numpy as np
import cv2
import os
from track_20_10_25 import track_data
TRACK_NAME = 'track_20_10_25'

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sort_tracker = Sort(max_age=3, min_hits=1, iou_threshold=0.3)
deepsort_tracker = DeepsortTracker(model_path="mars-small128.pb")

if __name__ == '__main__':
    os.makedirs(os.path.join("TrackEval/data/gt/mot_challenge/country_ball-train", TRACK_NAME, 'gt'), exist_ok=True)
    os.makedirs(os.path.join("TrackEval/data/trackers/mot_challenge/country_ball-train/data/data"), exist_ok=True)

    with open(os.path.join("TrackEval/data/gt/mot_challenge/country_ball-train", TRACK_NAME, "seqinfo.ini"), 'w') as f:
        f.write('[Sequence]\n')
        f.write(f"seqLength={len(track_data)}")
    gt_file = open(os.path.join("TrackEval/data/gt/mot_challenge/country_ball-train", TRACK_NAME, "gt/gt.txt"), 'w')
    sort_tracks = open(os.path.join("TrackEval/data/trackers/mot_challenge/country_ball-train/data/data", f'{TRACK_NAME}_sort.txt'), 'w')
    deep_sort_tracks = open(os.path.join("TrackEval/data/trackers/mot_challenge/country_ball-train/data/data", f'{TRACK_NAME}_deep_sort.txt'), 'w')
    for el in track_data:
        # gt
        for track in el['data']:
            if not track['bounding_box']:
                continue
            gt_file.write(f"{el['frame_id']}, {track['cb_id']}, {', '.join(map(lambda x: str(max(0, x)),track['bounding_box']))}, -1, -1, -1, -1\n")

        # SORT
        dets = np.array(list([[*det['bounding_box'], 1, i] for i, det in enumerate(el['data']) if det['bounding_box']]))
        if dets.size != 0:
            tracked_dets = sort_tracker.update(dets)
            for track in tracked_dets:
                sort_tracks.write(f"{el['frame_id']}, {int(track[-2])}, {', '.join(map(str, track[:4]))}, -1, -1, -1, -1\n")

        # Deep SORT
        dets = list([ObjDetection(np.array([max(det['bounding_box'][0], 0),
                                            max(det['bounding_box'][1], 0),
                                            max(det['bounding_box'][2] - det['bounding_box'][0], 0),
                                            max(det['bounding_box'][3] - det['bounding_box'][1], 0)]),
                                  1, i)
                     for i, det in enumerate(el['data']) if det['bounding_box']])
        frame = cv2.imread(os.path.join(TRACK_NAME, f'{el["frame_id"]}.png'))
        result = copy.deepcopy(el)
        if dets:
            tracked_dets = deepsort_tracker.track_boxes(frame, dets)
            for track in tracked_dets:
                deep_sort_tracks.write(
                    f"{el['frame_id']}, {int(track.tracking_id)}, {track.x_min}, {track.y_min}, "
                    f"{track.x_max}, {track.y_max}, -1, -1, -1, -1\n")
    gt_file.close()
    shutil.copytree(os.path.join("TrackEval/data/gt/mot_challenge/country_ball-train", TRACK_NAME),
                    os.path.join("TrackEval/data/gt/mot_challenge/country_ball-train", f"{TRACK_NAME}_sort"), symlinks=True, dirs_exist_ok=True)
    shutil.copytree(os.path.join("TrackEval/data/gt/mot_challenge/country_ball-train", TRACK_NAME),
                    os.path.join("TrackEval/data/gt/mot_challenge/country_ball-train", f"{TRACK_NAME}_deep_sort"), symlinks=True, dirs_exist_ok=True)