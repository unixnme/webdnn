from typing import Optional

from webdnn.graph.axis import Axis
from webdnn.graph.operator import Operator
from webdnn.graph.operators.attributes.tensorwise import Tensorwise
from webdnn.graph.placeholder import IntOrPlaceholder
from webdnn.graph.variable import Variable


class Resize2D(Operator):
    """Resize2D(name, axis1, size1, axis2, size2)
    Resize Tensor by nearest neighbor algorithm.

    Args:
        name (str): Operator name.
        axis1 (Axis): resize axis 1
        size1 (int): output size of `axis1`
        axis2 (Axis): resize axis 2
        size2 (int): output size of `axis2`

    Signature
        .. code::

            y, = op(x)

        - **x** - Input variable.
        - **y** - Output variable.
    """

    def __init__(self, name: Optional[str],
                 axis1: Axis, size1: int,
                 axis2: Axis, size2: int):
        super().__init__(name)
        self.parameters["axis1"] = axis1
        self.parameters["axis2"] = axis2
        self.parameters["size1"] = size1
        self.parameters["size2"] = size2
        assert axis1 != axis2

    def __call__(self, x: Variable):
        self.append_input("x", x)
        assert self.axis1 in x.order.axes, ValueError(f"x doesn't has {self.axis1}")
        assert self.axis2 in x.order.axes, ValueError(f"x doesn't has {self.axis2}")

        for axis in x.order.axes:
            if axis == self.axis1 or self.axis2:
                continue
            self.attributes.add(Tensorwise(self, axis))

        return self.exec()

    def exec(self):
        x = self.inputs["x"]

        y_shape = list(x.shape)
        y_shape[x.order.axes_dict[self.axis1]] = self.size1
        y_shape[x.order.axes_dict[self.axis2]] = self.size2

        y = Variable(y_shape, x.order)

        self.append_output("y", y)
        return y,

    @property
    def axis1(self) -> Axis:
        return self.parameters["axis1"]

    @property
    def axis2(self) -> Axis:
        return self.parameters["axis2"]

    @property
    def size1(self) -> IntOrPlaceholder:
        return self.parameters["size1"]

    @property
    def size2(self) -> IntOrPlaceholder:
        return self.parameters["size2"]
