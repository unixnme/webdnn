from typing import Optional, Sequence, Tuple

from webdnn.graph.graph import Graph
from webdnn.graph.operator import Operator
from webdnn.graph.optimize_rule import OptimizeRule
from webdnn.graph.variable import Variable
from webdnn.graph.variables.constant_variable import ConstantVariable


class Slice(Operator):
    """Slice(name)
    Slice input variable. This operator cannot be remove nor insert axes.

    Args:
        name (str): Operator name.
        begin (tuple of int): Position of start slicing
        end (tuple of int): Position of start slicing
        stride (tuple of int): Position of start slicing

    Signature
        .. code::

            y, = op(x)

        - **x** - Input variable.
        - **y** - Output variable.

    """

    def __init__(self, name: Optional[str], begin: Sequence[int], end: Sequence[int], stride: Sequence[int]):
        super().__init__(name)
        self.parameters["begin"] = begin
        self.parameters["end"] = end
        self.parameters["stride"] = stride

    def __call__(self, x: Variable):
        self.append_input("x", x)
        return self.exec()

    def exec(self):
        x = self.inputs["x"]
        y_shape = tuple((e - b) // s for b, e, s in zip(self.begin, self.end, self.stride))
        y = Variable(y_shape, x.order)
        self.append_output("y", y)
        return y,

    @property
    def begin(self) -> Tuple[int]:
        return tuple(self.parameters["begin"])

    @property
    def end(self) -> Tuple[int]:
        return tuple(self.parameters["end"])

    @property
    def stride(self) -> Tuple[int]:
        return tuple(self.parameters["stride"])

    def fold_constance(self, graph: Graph):
        x = self.inputs["x"]  # type: ConstantVariable
        y = self.outputs["y"]
        indices = tuple(slice(b, e, s) for b, e, s in zip(self.begin, self.end, self.stride))

        new_y = ConstantVariable(x.data[indices], x.order)
        new_y.change_order(y.order)
        OptimizeRule.replace_variable(graph, y, new_y)
        self.remove_all()
