import logging
from typing import Dict, List, Optional

import numpy as np

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.import_utils import check_requirements

logger = logging.getLogger(__name__)


class YoloxOnnxDetectionModel(DetectionModel):
    def check_dependencies(self) -> None:
        check_requirements(["onnxruntime"])

    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """
        from yolox.yolox_onnx import YoloxONNX

        # Load config
        with open(self.config_path) as fp:
            import json
            config = json.load(fp)

        self.confidence_threshold = config['class_score_th']

        # set model
        try:
            self.model = YoloxONNX(
                model_path=self.model_path,
                input_shape=[int(i) for i in config['input_shape'].split(',')],
                class_score_th=self.confidence_threshold,
                nms_th=config['nms_th'],
                nms_score_th=config['nms_score_th'],
                with_p6=config['with_p6'],
                device=self.device,
            )
        except Exception as e:
            TypeError("model_path is not a valid yolox model path: ", e)

        # set category list
        self.category_name_list = list(self.category_mapping.values())
        self.category_name_list_len = len(self.category_name_list)

    def perform_inference(self, image: np.ndarray, image_size: int = None):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in BGR order.
            image_size: int
                Inference input size.
        """

        # Confirm model is loaded
        assert self.model is not None, "Model is not loaded, load it by calling .load_model()"

        prediction_result = self.model.inference(image)

        self._original_predictions = [prediction_result]

    @property
    def num_categories(self):
        return self.category_name_list_len

    @property
    def has_mask(self):
        return False

    @property
    def category_names(self):
        return self.category_name_list

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions

        # compatilibty for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        # handle all predictions
        object_prediction_list_per_image = []
        for image_ind, original_prediction in enumerate(original_predictions):
            bboxes = original_prediction[0]
            scores = original_prediction[1]
            class_ids = original_prediction[2]

            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[
                image_ind]
            object_prediction_list = []

            # process predictions
            for original_bbox, score, class_id in zip(bboxes, scores,
                                                      class_ids):
                x1 = int(original_bbox[0])
                y1 = int(original_bbox[1])
                x2 = int(original_bbox[2])
                y2 = int(original_bbox[3])
                bbox = [x1, y1, x2, y2]
                score = score
                category_id = int(class_id)
                category_name = self.category_mapping[str(category_id)]

                # fix negative box coords
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = max(0, bbox[2])
                bbox[3] = max(0, bbox[3])

                # fix out of image box coords
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                # ignore invalid predictions
                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    logger.warning(
                        f"ignoring invalid prediction with bbox: {bbox}")
                    continue

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    bool_mask=None,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image
