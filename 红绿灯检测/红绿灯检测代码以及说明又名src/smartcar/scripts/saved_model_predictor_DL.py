#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.predictor.saved_model_predictor import _get_signature_def
from tensorflow.contrib.predictor.saved_model_predictor import _check_signature_arguments
from tensorflow.contrib.predictor.predictor import Predictor
import tensorflow as tf

class SavedModelPredictor(Predictor):
  """A `Predictor` constructed from a `SavedModel`."""

  def __init__(self,
               export_dir,
               signature_def_key=None,
               signature_def=None,
               input_names=None,
               output_names=None,
               tags=None,
               graph=None):
    _check_signature_arguments(
        signature_def_key, signature_def, input_names, output_names)
    tags = tags or "serve"
    self._graph = graph or tf.get_default_graph()

    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_options)

    with self._graph.as_default():   
      self._session = tf.Session(config=config)
      tf.saved_model.loader.load(self._session, tags.split(','), export_dir)

    if input_names is None:
      if signature_def is None:
        signature_def = _get_signature_def(signature_def_key, export_dir, tags)
      input_names = {k: v.name for k, v in signature_def.inputs.items()}
      output_names = {k: v.name for k, v in signature_def.outputs.items()}

    self._feed_tensors = {k: self._graph.get_tensor_by_name(v)
                          for k, v in input_names.items()}
    self._fetch_tensors = {k: self._graph.get_tensor_by_name(v)
                           for k, v in output_names.items()}
