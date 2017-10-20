from typing import Dict, List

from webdnn.backend.code_generator.injectors.kernel_name_injector import KernelNameInjector
from webdnn.backend.webgl.generator import WebGLDescriptorGenerator
from webdnn.backend.webgl.kernel import Kernel
from webdnn.backend.webgl.kernels.util import texture_stride, texture_shape, FragmentShaderPreamble, simplify_orders
from webdnn.backend.webgl.uniform_injector import UniformInjector
from webdnn.graph.axis import Axis
from webdnn.graph.axis import AxisKeyDict
from webdnn.graph.operators.resize_2d import Resize2D
from webdnn.graph.order import Order
from webdnn.graph.variable import Variable
from webdnn.util.misc import mul

template = FragmentShaderPreamble + """
%%UNIFORM(vec2, texture_stride_y)%%;
%%UNIFORM(vec4, variable_shape_y)%%;
%%UNIFORM(vec4, variable_stride_y)%%;

%%UNIFORM(sampler2D, sampler_x)%%;
%%UNIFORM(vec2, texture_shape_x)%%;
%%UNIFORM(vec2, texture_stride_x)%%;
%%UNIFORM(vec4, variable_shape_x)%%;
%%UNIFORM(vec4, variable_stride_x)%%;

void main() {
    vec4 variable_position_y = convert_position(gl_FragCoord.xy, texture_stride_y, variable_stride_y, variable_shape_y);
    vec4 variable_position_x = variable_position_y * variable_shape_x / variable_shape_y + 0.5;  
    vec2 texture_coord_x = convert_coord(variable_position_x, variable_stride_x, texture_stride_x, texture_shape_x);
    gl_FragColor = texture2D(sampler_x, texture_coord_x);
}
"""


def _optimize_loop_structure(variables: List[Variable], key_variable: Variable, keep_axes: List[Axis] = None):
    """
    Optimize loop structure to iterate each element in variables

    Returns:
        (tuple): two elements are returned

        - First one is shape dictionary of all variables.
        - Second one is stride dictionary of all variables.
    """
    orders, shape_dicts = simplify_orders(variables,
                                          keep_axes=keep_axes)  # type: Dict[Variable, Order], Dict[Variable, AxisKeyDict[List[int]]]
    shapes = {v: [shape_dicts[v][a] for a in orders[v].axes] for v in variables}
    strides = {v: [mul(shapes[v][orders[v].axes_dict[a] + 1:]) for a in orders[v].axes] for v in variables}
    stride_dicts = {v: AxisKeyDict(orders[v].axes, strides[v]) for v in variables}

    # Re-ordering shapes and strides along to key variable's order
    axes = []
    axes += [axis for axis in orders[key_variable].axes if axis not in axes]
    for v in sorted(variables, key=lambda v: orders[v].ndim):
        axes += [axis for axis in orders[v].axes if axis not in axes]

    orders = {v: Order(list(filter(lambda x: x in orders[v].axes, axes))) for v in variables}

    key_order = orders[key_variable]
    shapes = {v: [shape_dicts[v][a] if a in orders[v].axes else 1 for a in key_order.axes] for v in variables}
    strides = {v: [stride_dicts[v][a] if a in orders[v].axes else 1 for a in key_order.axes] for v in variables}

    # Padding shapes and strides to 4D
    if key_order.ndim > 4:
        raise NotImplementedError(f"Too large number of dimension: {v}")

    for v in variables:
        shape = shapes[v]
        stride = strides[v]
        while len(shape) < 4:
            stride.append(1)
            shape.append(1)

    return shapes, strides


@WebGLDescriptorGenerator.register_handler(Resize2D)
def resize2D(op: Resize2D):
    x = op.inputs["x"]
    y = op.outputs["y"]

    shapes, strides = _optimize_loop_structure([x, y], y)
    name_injector = KernelNameInjector(op)
    uniform_injector = UniformInjector()

    uniform_injector.register({
        f"sampler_x": x,

        "texture_stride_y": texture_stride(y),
        "variable_shape_y": shapes[y],
        "variable_stride_y": strides[y],

        f"texture_shape_x": texture_shape(x),
        f"texture_stride_x": texture_stride(x),
        f"variable_shape_x": shapes[x],
        f"variable_stride_x": strides[x],
    })

    source = template
    source = uniform_injector.inject(source)
    source = name_injector.inject(source)
    kernel = Kernel(
        source,
        name_injector.name,
        uniform_injector.samplers,
        uniform_injector.uniforms,
        y
    )

    return [kernel]
