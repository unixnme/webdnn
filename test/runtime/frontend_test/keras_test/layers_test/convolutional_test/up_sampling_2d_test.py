import numpy as np

from test.runtime.frontend_test.keras_test.util import keras, KerasConverter
from test.util import generate_kernel_test_case, wrap_template


@wrap_template
def template(shape=(14, 15, 4), description: str = ""):
    x = keras.layers.Input(shape)
    y = keras.layers.UpSampling2D(size=2)(x)
    model = keras.models.Model([x], [y])

    vx = np.arange(2*np.prod(shape)).reshape(2, *shape).astype(np.float32)
    vy = model.predict(vx, batch_size=2)

    graph = KerasConverter(batch_size=2).convert(model)

    generate_kernel_test_case(
        description=f"[keras] UpSampling2D {description}",
        graph=graph,
        backend=["webgpu", "webgl", "webassembly"],
        inputs={graph.inputs[0]: vx},
        expected={graph.outputs[0]: vy},

        # TODO: replace computation algorithm with more accurate one
        EPS=1e-2
    )


def test():
    template()
