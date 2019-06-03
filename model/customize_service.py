import ast
import io

from PIL import Image
import numpy as np
import tensorflow as tf
import numpy as np
import h5py
import time
import os
from model_service.tfserving_model_service import TfServingBaseService

class object_detection_service(TfServingBaseService):
  def _preprocess(self, data):
    preprocessed_data = {}
    for k, v in data.items():
      for file_name, file_content in v.items():
        image = Image.open(file_content)
        image = image.convert('RGB')
        image = np.asarray(image, dtype=np.float32)
        image = image[np.newaxis, :, :, :]
        preprocessed_data[k] = image
    return preprocessed_data

  def _postprocess(self, data):
    h5f = h5py.File(os.path.join(self.model_path, 'index'), 'r')
    labels_list = h5f['labels_list'][:]
    h5f.close()
    num_boxes = len(data['detection_classes'])
    classes = []
    boxes = []
    scores = []
    prob_threshold = 0.3
    result_return = dict()
    for i in range(num_boxes):
      if data['detection_scores'][i] > prob_threshold:
        class_id = data['detection_classes'][i] - 1
        classes.append(labels_list[int(class_id)])
        boxes.append(data['detection_boxes'][i])
        scores.append(data['detection_scores'][i])
    ##########add NMS#######################################
    nms_iou_threshold = 0.3
    bounding_boxes = boxes
    confidence_score = scores
      # Bounding boxes
    boxes = np.array(bounding_boxes)
    picked_boxes = []
    picked_score = []
    picked_classes = []
    if len(boxes) != 0:
      # coordinates of bounding boxes
      start_x = boxes[:, 0]
      start_y = boxes[:, 1]
      end_x = boxes[:, 2]
      end_y = boxes[:, 3]
      # Confidence scores of bounding boxes
      score = np.array(confidence_score)
      # Picked bounding boxes
      # Compute areas of bounding boxes
      areas = (end_x - start_x + 1) * (end_y - start_y + 1)
      # Sort by confidence score of bounding boxes
      order = np.argsort(score)
      # Iterate bounding boxes
      while order.size > 0:
        # The index of largest confidence score
        index = order[-1]
        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        picked_classes.append(classes[index])
        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])
        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h
        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)
        left = np.where(ratio < nms_iou_threshold)
        order = order[left]

    result_return['detection_classes'] = picked_classes
    result_return['detection_boxes'] = picked_boxes
    result_return['detection_scores'] = picked_score
    return result_return

'''
class customized_converter(object):
  def pre_processing(self, input_data, model_inputs_schema, request):
    for k, v in input_data.items():
      if isinstance(v, dict):
        is_images = False
        for file_name, file_content in v.items():
          if isinstance(file_content, io.BytesIO):
            is_images = True

        if is_images:
          for file_name, file_content in v.items():
            image = Image.open(file_content) 
            image = np.asarray(image, dtype=np.float32)
            image = image[np.newaxis, :, :, :] 
          request.inputs[k].CopyFrom(make_tensor_proto(image, model_inputs_schema[k]))
        else:
          request.inputs[k].CopyFrom(make_tensor_proto(v, model_inputs_schema[k]))
      else:
        request.inputs[k].CopyFrom(make_tensor_proto(v, model_inputs_schema[k]))
    return request

  def post_processing(self, predict_response):
    # load the index file (which should be included in the package delivered to uPredict) 
    # storing the label_map_dict
    current_dir = os.path.dirname(os.path.realpath("__file__"))
    h5f = h5py.File(os.path.join(current_dir, 'model', '1', 'index'), 'r')
    label_name_list = h5f['label_name_list'][:]
    h5f.close()
    result = dict()
    for k in ['detection_classes', 'detection_boxes', 'detection_scores']:
      tensor_proto = predict_response.outputs[k]
      result[k] = tf.contrib.util.make_ndarray(tensor_proto).tolist()
    threshold = tf.contrib.util.make_ndarray(predict_response.outputs['threshold']).tolist()[0]
    num_boxes = len(result['detection_classes'])
    classes = []
    boxes = []
    scores = []
    threshold = 0.8
    result_return = dict()
    for i in range(num_boxes):
      if result['detection_scores'][i] > threshold:
        class_id = result['detection_classes'][i] - 1
        classes.append(label_name_list[int(class_id)])
        boxes.append(result['detection_boxes'][i])
        scores.append(result['detection_scores'][i])
    result_return['detection_classes'] = classes
    result_return['detection_boxes'] = boxes
    result_return['detection_scores'] = scores
    return result_return
'''



# predict_object
# {u'image': tf.float32}
# {u'detection_classes': tf.float32, u'detection_boxes': tf.float32, u'detection_scores': tf.float32}
