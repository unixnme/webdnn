# -*- coding:utf-8 -*-

from typing import List

from webdnn.frontend.converter import Converter
from webdnn.frontend.tensorflow import TensorFlowConverter
from webdnn.graph.graph import Graph
from webdnn.graph.order import Order
from webdnn.util import console

FLAG_KERAS_INSTALLED = False

try:
    import keras
    import keras.backend as K
    import tensorflow as tf

    if not "2." <= keras.__version__ < "3.":
        console.debug(f"WebDNN supports Keras v2.*.*. Currently, keras {keras.__version__} is installed.")
    FLAG_KERAS_INSTALLED = True

except ImportError as e:
    console.debug("Keras and Tensorflow are not completely installed.")
    pass


class KerasConverter(Converter["keras.layers.Layer"]):
    """KerasConverter(batch_size=1)

    Convert keras.models.model into WebDNN IR.

    **Limitations**

    - Only Keras v2+ is supported.
    - Only tensorflow backend is supported.
    - Only :code:`data_format="channel_last"` is supported.

    If you want to implement custom handler for your custom Keras Layer, please see :doc:`/tutorial/custom_operator/index`.

    Args:
        batch_size(int or None): input batch size. As default, keras handle the batch size as place holder (undetermined) value. If
          :code:`None` is passed, converter handles the batch size as placeholder named "N".
    """

    def __init__(self, batch_size: int = 1):
        if not FLAG_KERAS_INSTALLED:
            raise ImportError("ImportError is occurred when Keras and Tensorflow are loaded.")

        if K.backend() != "tensorflow":
            raise NotImplementedError("Only TensorFlow backend is supported.")

        self._session = K.get_session()
        self._batch_size = batch_size
        self._tf_converter = TensorFlowConverter(session=self._session, batch_size=self._batch_size)

    def convert(self, model: "keras.models.Model", input_orders: List[Order] = None) -> Graph:
        """convert(model, input_orders=None)

        Convert kerasmodel into WebDNN IR Graph.

        Args:
            model (`keras.models.Model`): keras model
            input_orders (list of :class:`~webdnn.graph.order.Order`): Order of input tensors. If `None` is passed, default order
                (`OrderNC` for 2D, `OrderNTC` for 3D, `OrderNHWC` for 4D) is used. If `input_orders=None`, default orders
                are assigned to all input tensors. If `input_orders[0]=None`, only first input tensor are converted with
                the default order.

        .. admonition:: Example

            .. code::

                model = keras.models.load_model("pre_trained_model.h5")
                graph = KerasConverter(batch_size=1).convert(model)

        Returns:
            (:class:`~webdnn.graph.graph.Graph`): WebDNN IR Graph
        """
        return self._tf_converter.convert(model.input, model.output)
