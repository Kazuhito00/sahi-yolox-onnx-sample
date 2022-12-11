#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2

from sahi.models.yolox_onnx import YoloxOnnxDetectionModel
from sahi.predict import get_sliced_prediction


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument(
        "--model",
        type=str,
        default='yolox/model/yolox_nano.onnx',
    )
    parser.add_argument(
        "--config",
        type=str,
        default='yolox/config.json',
    )

    parser.add_argument("--slice_height", type=int, default=512)
    parser.add_argument("--slice_width", type=int, default=512)
    parser.add_argument("--overlap_height_ratio", type=float, default=0.2)
    parser.add_argument("--overlap_width_ratio", type=float, default=0.2)

    parser.add_argument("--draw_score_th", type=float, default=0.3)

    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    if args.movie is not None:
        cap_device = args.movie

    model_path = args.model
    config_path = args.config

    slice_height = args.slice_height
    slice_width = args.slice_width
    overlap_height_ratio = args.overlap_height_ratio
    overlap_width_ratio = args.overlap_width_ratio

    draw_score_th = args.draw_score_th

    # カメラ準備 ###############################################################
    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    # COCOクラスリスト読み込み
    with open('coco_classes.txt', 'rt') as f:
        coco_classes = f.read().rstrip('\n').split('\n')
        category_mapping = {
            str(ind): category_name
            for ind, category_name in enumerate(coco_classes)
        }

    # モデルロード #############################################################
    yolox = YoloxOnnxDetectionModel(
        model_path=model_path,
        config_path=config_path,
        category_mapping=category_mapping,
        device="cpu",  # or 'cuda:0'
    )

    while True:
        start_time = time.time()

        # カメラキャプチャ ################################################
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # 推論実施 ########################################################
        result = get_sliced_prediction(
            frame,
            yolox,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            verbose=0,
        )

        bboxes = []
        scores = []
        class_ids = []
        class_names = []
        for temp in result.object_prediction_list:
            bboxes.append([
                temp.bbox.minx,
                temp.bbox.miny,
                temp.bbox.maxx,
                temp.bbox.maxy,
            ])
            scores.append(temp.score.value)
            class_ids.append(temp.category.id)
            class_names.append(temp.category.name)

        elapsed_time = time.time() - start_time

        # デバッグ描画
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            bboxes,
            scores,
            class_ids,
            class_names,
            draw_score_th,
        )

        # キー処理(ESC：終了) ##############################################
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映 #########################################################
        cv2.imshow('SAHI YOLOX Sample', debug_image)

    cap.release()
    cv2.destroyAllWindows()


def get_id_color(index):
    temp_index = abs(int(index + 1)) * 3
    color = ((37 * temp_index) % 255, (17 * temp_index) % 255,
             (29 * temp_index) % 255)
    return color


def draw_debug(
    image,
    elapsed_time,
    bboxes,
    scores,
    class_ids,
    class_names,
    draw_score_th,
):
    debug_image = copy.deepcopy(image)

    for bbox, score, class_id, class_name in zip(
            bboxes,
            scores,
            class_ids,
            class_names,
    ):
        if draw_score_th > score:
            continue

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        color = get_id_color(class_id)

        # バウンディングボックス
        debug_image = cv2.rectangle(
            debug_image,
            (x1, y1),
            (x2, y2),
            color,
            thickness=2,
        )

        # クラスID、スコア
        score = '%.2f' % score
        text = '%s:%s' % (class_name, score)
        debug_image = cv2.putText(
            debug_image,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            thickness=2,
        )

    # 推論時間
    text = 'Elapsed time:' + '%.0f' % (elapsed_time * 1000)
    text = text + 'ms'
    debug_image = cv2.putText(
        debug_image,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        thickness=2,
    )

    return debug_image


if __name__ == '__main__':
    main()
