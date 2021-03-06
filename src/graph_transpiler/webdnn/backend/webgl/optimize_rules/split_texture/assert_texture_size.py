from webdnn.backend.webgl.attributes.channel_mode import ChannelMode
from webdnn.backend.webgl.attributes.texture_shape import TextureShape
from webdnn.graph import traverse
from webdnn.graph.graph import Graph
from webdnn.graph.optimize_rule import OptimizeRule
from webdnn.util import config


class AssertTextureSize(OptimizeRule):
    def optimize(self, graph: Graph):
        traverse.dump(graph)
        MAX_SIZE = config.WEBGL_MAX_TEXTURE_SIZE

        for v in traverse.listup_variables(graph):
            height, width = TextureShape.get(v)
            assert height <= MAX_SIZE and width <= MAX_SIZE, f"Texture size is invalid: {v.name} \n" \
                                                             f"  (variable shape)={v.shape}, \n" \
                                                             f"  (channel mode)={ChannelMode.get(v).name}, \n" \
                                                             f"  (texture shape)=(width={width}, height={height}), \n" \
                                                             f"  (WEBGL_MAX_TEXTURE_SIZE)={config.WEBGL_MAX_TEXTURE_SIZE}"
        return graph, False
