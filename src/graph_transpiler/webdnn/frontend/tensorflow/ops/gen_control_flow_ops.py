import numpy as np
import tensorflow as tf

from webdnn.frontend.constraints import AxisVar
from webdnn.frontend.tensorflow.converter import TensorFlowConverter
from webdnn.graph.order import Order
from webdnn.graph.variables.constant_variable import ConstantVariable


@TensorFlowConverter.register_handler("Abort")
def abort_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("ControlTrigger")
def control_trigger_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("Enter")
def enter_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("Exit")
def exit_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("LoopCond")
def loop_cond_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("Merge")
def merge_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    for i, tf_tensor in enumerate(tf_op.inputs):
        if converter.has_variable(tf_tensor):
            converter.set_variable(tf_op.outputs[0], converter.get_variable(tf_tensor))
            # noinspection PyTypeChecker
            converter.set_variable(tf_op.outputs[1], ConstantVariable(np.array([i], dtype=np.float32), Order([AxisVar()])))
            break

    else:
        raise ValueError(f"[TensorFlowConverter] 'Merge' operator is called without any resolved tensor")


@TensorFlowConverter.register_handler("NextIteration")
def next_iteration_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("NoOp")
def no_op_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("RefEnter")
def ref_enter_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("RefExit")
def ref_exit_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("RefMerge")
def ref_merge_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("RefNextIteration")
def ref_next_iteration_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("RefSelect")
def ref_select_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("RefSwitch")
def ref_switch_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    raise NotImplementedError(f"[TensorFlowConverter] {tf_op.type} is not supported yet.")


@TensorFlowConverter.register_handler("Switch")
def switch_handler(converter: TensorFlowConverter, tf_op: "tf.Operation"):
    data = converter.get_variable(tf_op.inputs[0])
    pred = converter.get_variable(tf_op.inputs[1])

    assert isinstance(pred, ConstantVariable), NotImplementedError(
        f"[TensorFlowConverter] 'Switch' operator with dynamic condition is not supported.")

    pred = pred.data
    if pred.flatten()[0]:
        converter.set_variable(tf_op.outputs[0], data)
    else:
        converter.set_variable(tf_op.outputs[1], data)
