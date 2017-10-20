# -*- coding:utf-8 -*-
from typing import List, Set, Union, Optional, Dict

from webdnn.frontend.constraints import unify, AxisVar
from webdnn.frontend.converter import Converter
from webdnn.graph.axis import Axis
from webdnn.graph.graph import Graph
from webdnn.graph.optimize_rule import OptimizeRuleGroup
from webdnn.graph.order import Order
from webdnn.graph.placeholder import Placeholder
from webdnn.graph.variable import Variable
from webdnn.graph.variables.attributes.input import Input
from webdnn.graph.variables.attributes.output import Output
from webdnn.graph.variables.constant_variable import ConstantVariable
from webdnn.optimizer.sub_rules.constant_folding import ConstantFolding
from webdnn.optimizer.tensorflow_frontend_optimize_rule import TensorFlowFrontendOptimizeRule
from webdnn.util import console
from webdnn.util import flags

FLAG_TF_INSTALLED = True

try:
    # noinspection PyPackageRequirements, PyUnresolvedReferences
    import tensorflow as tf

except ImportError as e:
    console.debug("Tensorflow are not completely installed.")
    FLAG_TF_INSTALLED = False
    pass


class CyclicGraphError(Exception):
    pass


class TensorFlowConverter(Converter["tf.Operation"]):
    """TensorFlowConverter(batch_size=1)

    Convert tf.Graph into WebDNN IR.

    Args:
        session (:code:`tf.Session`): Session

    """

    def __init__(self, session: "tf.Session", batch_size: int = 1):
        super(TensorFlowConverter, self).__init__()

        if not FLAG_TF_INSTALLED:
            raise ImportError("ImportError is occurred when Tensorflow is loaded.")

        self.session = session
        self._batch_size = Placeholder(label=Axis.N.name, value=batch_size)

    def serialize_operator_type(self, op: "tf.Operation") -> str:
        return op.type

    def convert(self, inputs: List["tf.Tensor"], outputs: List["tf.Tensor"],
                order_hints: Optional[Dict[Union["tf.Tensor", "tf.Variable"], Order]] = None) -> Graph:
        """convert(model, input_orders=None)

        Args:
            inputs (list of `tf.Tensor`): tensorflow input tensors
            outputs (list of `tf.Tensor`): tensorflow output tensors
            order_hints: Order annotations which helps webdnn's optimizer.

        .. admonition:: Example

            .. code::

                # y = x @ W + b
                x = tf.placeholder(tf.float32, [None, 784])
                W = tf.Variable(tf.zeros([784, 10]))
                b = tf.Variable(tf.zeros([10]))
                y = tf.nn.softmax(tf.matmul(x, W) + b)

                webdnn_graph = TensorFlowConverter().convert([x], [y])

        Returns:
            (:class:`~webdnn.graph.graph.Graph`): WebDNN IR Graph
        """

        for tensor in inputs:
            shape = [Placeholder() if dim.value is None else dim.value for dim in tensor.shape.dims]
            if isinstance(shape[0], Placeholder):
                shape[0] = self._batch_size
            # noinspection PyTypeChecker
            self.set_variable(tensor, Variable(shape, Order([AxisVar() for _ in shape])))

        ops = _listup_operations(inputs, outputs)
        for op in ops:
            self._convert_operator(op)
            sub_graph = Graph([self.get_variable(tf_tensor) for tf_tensor in op.inputs if self.has_variable(tf_tensor)],
                              [self.get_variable(tf_tensor) for tf_tensor in op.outputs if self.has_variable(tf_tensor)])

            # Constant folding improves possibility of conversion, because many tensors are used not only for main input variable but also
            # for other parameter like indices of operation, and WebDNN doesn't support dynamic indices operation.
            OptimizeRuleGroup([ConstantFolding()], repeat=True).optimize(sub_graph)
            for tf_tensor, webdnn_output in zip(op.outputs, sub_graph.outputs):
                if self.get_variable(tf_tensor) != webdnn_output:
                    self.set_variable(tf_tensor, webdnn_output, overwrite=True)

        if order_hints:
            for tensor, order in order_hints.items():
                if isinstance(tensor, tf.Variable):
                    tensor = tensor.value()

                variable = self.get_variable(tensor)
                for axis1, axis2 in zip(variable.order.axes, order.axes):
                    unify(axis1, axis2)

        if flags.AGGRESSIVE_ORDER_INFERENCE:
            # 1st dimension of output variable is batch size
            for tensor in outputs:
                variable = self.get_variable(tensor)
                unify(variable.order.axes[0], Axis.N)

        # Remove redundant ReinterpretAxis operators
        graph = Graph([self.get_variable(tensor) for tensor in inputs], [self.get_variable(tensor) for tensor in outputs])
        graph, _ = TensorFlowFrontendOptimizeRule().optimize(graph)

        for v in graph.inputs:
            v.attributes.add(Input(v))

        for v in graph.outputs:
            v.attributes.add(Output(v))

        return graph

    def convert_to_constant_variable(self, tensor: "tf.Tensor", order: Optional[Order] = None) -> ConstantVariable:
        """convert_to_constant_variable(tf_var, order)

        Convert tf.Tensor into :class:`~webdnn.graph.variables.constant_variable.ConstantVariable`.

        This method also registers the mapping information between TensorFlow variable and WebDNN constant variable.
        If specified TensorFlow variable is already registered into converter, converter checks that the shape and order
        is valid

        **This method is provided only for implementing custom converter handler.**

        Args:
            tensor (:code:`tf.Tensor`): TensorFlow tensor
            order: (:class:`~webdnn.graph.order.Order`) data order. As default, default order is used.

        Returns:
            (:class:`~webdnn.graph.variables.constant_variable.ConstantVariable`): converted variable.
        """

        data, = self.session.run([tensor])

        if self.has_variable(tensor):
            variable = self.get_variable(tensor)
            assert variable.shape == tuple(data.shape), f"[TensorFlowConverter] {tensor} is already registered before, and " \
                                                        f"shape mismatch is detected: (registered shape)=" \
                                                        f"{variable.shape}, (given tensor's shape)=" \
                                                        f"{tensor.shape}"
            if order is not None:
                assert variable.order == order, f"[TensorFlowConverter] {tensor} is already registered before, and order " \
                                                f"mismatch is detected: (registered order)={variable.order}, (given " \
                                                f"order)={order}"

        else:
            if order is None:
                # noinspection PyTypeChecker
                order = Order([AxisVar() for _ in range(data.ndim)])

            variable = ConstantVariable(data, order)
            self.set_variable(tensor, variable)

        return variable


def _listup_operations(inputs, outputs):
    stack = list(outputs)  # type: List[Union[tf.Tensor, tf.Operation]]
    resolved = set(inputs)  # type: Set[Union[tf.Tensor, tf.Operation]]
    result = []  # type: List[tf.Operation]

    while len(stack) > 0:
        node = stack.pop()
        if node in resolved:
            continue

        prev_nodes = [node.op] if isinstance(node, tf.Tensor) else node.inputs
        unresolved_prevs = [prev_node for prev_node in prev_nodes if prev_node not in resolved]

        if len(unresolved_prevs) == 0:
            # TensorFlow allows cyclic graph (like RNN), but WebDNN doesn't.
            resolved.add(node)
            if isinstance(node, tf.Operation):
                result.append(node)

        else:
            # # TensorFlow allows cyclic graph (like RNN), but WebDNN doesn't.
            # if all(n in stack for n in unresolved_prevs):
            #     print(node)
            #     print("unresolved_prevs")
            #     print(unresolved_prevs)
            #     raise CyclicGraphError('[TensorFlowConverter] Cyclic graph is detected.')

            stack.append(node)
            stack += unresolved_prevs

    return result
