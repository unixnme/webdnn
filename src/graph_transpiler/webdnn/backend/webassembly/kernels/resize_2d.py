from typing import List

from webdnn.backend.code_generator.allocator import MemoryLayout
from webdnn.backend.code_generator.injectors.buffer_injector import BufferInjector
from webdnn.backend.code_generator.injectors.kernel_name_injector import KernelNameInjector
from webdnn.backend.webassembly.generator import WebassemblyDescriptorGenerator
from webdnn.backend.webassembly.kernel import Kernel
from webdnn.graph.operators.resize_2d import Resize2D

template_up_sampling = """
void %%FUNC_NAME%%(const int * %%META_BUFFER%%)
{
    const float *X = %%LOAD_BUFFER(up_sampling_2d_X)%%;
    float *Y = %%LOAD_BUFFER(up_sampling_2d_Y)%%;
    const int dim = %%LOAD_BUFFER(up_sampling_2d_dim)%%;
    const int D1 = %%LOAD_BUFFER(up_sampling_2d_D1)%%;
    const int D2 = %%LOAD_BUFFER(up_sampling_2d_D2)%%;
    const int S1_y = %%LOAD_BUFFER(up_sampling_2d_S1_y)%%;
    const int S2_y = %%LOAD_BUFFER(up_sampling_2d_S2_y)%%;
    const int S1_x = %%LOAD_BUFFER(up_sampling_2d_S1_x)%%;
    const int S2_x = %%LOAD_BUFFER(up_sampling_2d_S2_x)%%;
    const int *y_shape = %%LOAD_BUFFER(up_sampling_2d_y_shape)%%;
    const int *y_stride = %%LOAD_BUFFER(up_sampling_2d_y_stride)%%;
    const int *x_stride = %%LOAD_BUFFER(up_sampling_2d_x_stride_in_y)%%;
    const int y_size = %%LOAD_BUFFER(up_sampling_2d_y_size)%%;

    for (int gid = 0; gid < y_size; gid++) {
        int i_x = 0;
        for (int d = 0; d < dim; d++) {
            if (d == D1) 
            {
                i_x += ((gid / y_stride[d]) % y_shape[d]) * S1_x / S1_y * x_stride[d];
            } 
            else if (d == D2) 
            {
                i_x += ((gid / y_stride[d]) % y_shape[d]) * S2_x / S2_y * x_stride[d];
            }
            else 
            {
                i_x += ((gid / y_stride[d]) % y_shape[d]) * x_stride[d];
            }
        }
        Y[gid] = X[i_x];
    }
}
"""


@WebassemblyDescriptorGenerator.register_handler(Resize2D)
def up_sampling_2d(op: Resize2D, memory_layout: MemoryLayout) -> List[Kernel]:
    x = op.inputs["x"]
    y = op.outputs["y"]

    buffer_injector = BufferInjector()
    buffer_injector.register({
        "up_sampling_2d_X": memory_layout[x],
        "up_sampling_2d_Y": memory_layout[y],
        "up_sampling_2d_dim": x.ndim,
        "up_sampling_2d_D1": y.order.axes_dict[op.axis1],
        "up_sampling_2d_D2": y.order.axes_dict[op.axis2],
        "up_sampling_2d_S1_y": y.shape_dict[op.axis1],
        "up_sampling_2d_S2_y": y.shape_dict[op.axis2],
        "up_sampling_2d_S1_x": x.shape_dict[op.axis1],
        "up_sampling_2d_S2_x": x.shape_dict[op.axis2],
        "up_sampling_2d_y_size": y.size,
        "up_sampling_2d_y_shape": y.shape,
        "up_sampling_2d_y_stride": y.stride,
        "up_sampling_2d_x_stride_in_y": [x.stride_dict[a] for a in y.order.axes],
    })

    name_injector = KernelNameInjector(op)

    source = template_up_sampling
    source = buffer_injector.inject(source)
    source = name_injector.inject(source)

    kernel = Kernel(
        {name_injector.name: source},
        name_injector.name,
        buffer_injector.buffer,
        buffer_injector.unresolved_value_list
    )

    return [kernel]
