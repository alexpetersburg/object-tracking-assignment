from fastapi import FastAPI, WebSocket
from track_3 import track_data, country_balls_amount
import asyncio
import glob
from sort.tracker import Sort
from deep_sort.inference import ObjDetection, DeepsortTracker
import numpy as np
import cv2
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


sort_tracker = Sort(max_age=3, min_hits=1, iou_threshold=0.3)
deepsort_tracker = DeepsortTracker(model_path="mars-small128.pb")

if __name__ == '__main__':
    for el in track_data[:5]:
        dets = np.array(list([[*det['bounding_box'], 1, i] for i, det in enumerate(el['data']) if det['bounding_box']]))
        tracked_dets = sort_tracker.update(dets)
        for track in tracked_dets:
            el['data'][int(track[-1])]['track_id'] = int(track[-2])
        dets = list([ObjDetection(np.array([det['bounding_box'][0],
                                            det['bounding_box'][1],
                                            det['bounding_box'][2] - det['bounding_box'][0],
                                            det['bounding_box'][3] - det['bounding_box'][1]]),
                                  1, i)
                     for i, det in enumerate(el['data']) if det['bounding_box']])
        frame = cv2.imread(os.path.join('frames_3', f'{el["frame_id"]}.png'))
        if dets:
            tracked_dets = deepsort_tracker.track_boxes(frame, dets)
            for track in tracked_dets:
                el['data'][track.det_id]['track_id'] = int(track.tracking_id)
    print(el)
    exit()

app = FastAPI(title='Tracker assignment')
imgs = glob.glob('imgs/*')
country_balls = [{'cb_id': x, 'img': imgs[x % len(imgs)]} for x in range(country_balls_amount)]
print('Started')


def tracker_soft(el):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов

    Ограничения:
    - необходимо использовать как можно меньше ресурсов (представьте, что
    вы используете embedded устройство, например Raspberri Pi 2/3).
    -значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме
    """
    dets = list([[*det['bounding_box'], 1, i] for i, det in enumerate(el['data']) if det['bounding_box']])
    if dets:
        tracked_dets = sort_tracker.update(np.array(dets))
        for track in tracked_dets:
            el['data'][int(track[-1])]['track_id'] = int(track[-2])
    return el


def tracker_strong(el):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов, скриншоты прогона

    Ограничения:
    - вы можете использовать любые доступные подходы, за исключением
    откровенно читерных, как например захардкодить заранее правильные значения
    'track_id' и т.п.
    - значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме

    P.S.: если вам нужны сами фреймы, измените в index.html значение make_screenshot
    на true для первого прогона, на повторном прогоне можете читать фреймы из папки
    и по координатам вырезать необходимые регионы.
    TODO: Ужасный костыль, на следующий поток поправить
    """
    dets = list([ObjDetection(np.array([det['bounding_box'][0],
                                        det['bounding_box'][1],
                                        det['bounding_box'][2] - det['bounding_box'][0],
                                        det['bounding_box'][3] - det['bounding_box'][1]]),
                              1, i)
                 for i, det in enumerate(el['data']) if det['bounding_box']])
    frame = cv2.imread(os.path.join('frames_3', f'{el["frame_id"]}.png'))
    if dets:
        tracked_dets = deepsort_tracker.track_boxes(frame, dets)
        for track in tracked_dets:
            el['data'][track.det_id]['track_id'] = int(track.tracking_id)
    return el


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print('Accepting client connection...')
    await websocket.accept()
    # отправка служебной информации для инициализации объектов
    # класса CountryBall на фронте
    await websocket.send_text(str(country_balls))
    for el in track_data:
        await asyncio.sleep(0.5)
        # TODO: part 1
        # el = tracker_soft(el)
        # TODO: part 2
        el = tracker_strong(el)
        # отправка информации по фрейму
        await websocket.send_json(el)
    sort_tracker.reset()
    deepsort_tracker.reset()
    print('Bye..')
