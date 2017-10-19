from typing import Tuple

from webdnn.graph import traverse
from webdnn.graph.graph import Graph
from webdnn.graph.operators.greater import Greater
from webdnn.graph.operators.greater_equal import GreaterEqual
from webdnn.graph.operators.select import Select
from webdnn.graph.optimize_rule import OptimizeRule


class ReplaceSelect(OptimizeRule):
    """
    Replace :class:`Select` in some cases.
    """

    def optimize(self, graph: Graph) -> Tuple[Graph, bool]:
        flag_changed = False
        for op in traverse.filter_nodes(traverse.listup_operators(graph), Select):  # type: Select
            cond = op.inputs["x0"]
            x1 = op.inputs["x1"]
            x2 = op.inputs["x2"]
            y = op.outputs["y"]

            if isinstance(cond.output_from, (Greater, GreaterEqual)):
                # cond is matrix of {0, 1}

                op.remove_all()
                OptimizeRule.replace_variable(graph, y, x1 * cond + x2 * (1 - cond))
                flag_changed = True
                continue

        return graph, flag_changed
