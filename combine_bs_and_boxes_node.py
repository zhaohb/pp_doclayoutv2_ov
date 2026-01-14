import argparse

import openvino as ov

from pathlib import Path

from openvino.runtime.passes import Manager, MatcherPass, WrapType, Matcher
from openvino.runtime import opset11 as ops

class combine_bs_and_boxes_node(MatcherPass):
    def __init__(self, transpose_node_list, ov_model):
        MatcherPass.__init__(self)
        self.model_changed = False
        self.ov_model = ov_model

        param = WrapType("opset11.Concat")

        def callback(matcher: Matcher) -> bool:
            root = matcher.get_match_root()
            if root is None:
                return False
            root_output = matcher.get_match_value()
            for y in transpose_node_list:
                root_name = root.get_friendly_name()
                if root_name.find(y) != -1 :
                    new_result = ops.result(root_output, name=f'{root_name}' + '/sink_port_0')
                    self.ov_model.add_results([new_result])
                    self.model_changed = True
                    transpose_node_list.remove(y)

            return True

        self.register_matcher(Matcher(param,"combine_bs_and_boxes_node"), callback)

def main():
    parser = argparse.ArgumentParser(description="Combine batch size and boxes nodes for pp_doclayoutv2 model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="OpenVINO IR directory (.xml 文件)"
    )
    
    args = parser.parse_args()
    core = ov.Core()

    layer_before_reshape_names = [
        "Concat.253"
    ]

    model_path = Path(args.model_path)
    ov_model = core.read_model(args.model_path)
    original_results = ov_model.get_results()

    manager = Manager()
    manager.register_pass(combine_bs_and_boxes_node(layer_before_reshape_names, ov_model))
    manager.run_passes(ov_model)
    for _result in original_results:
        ov_model.remove_result(_result)

    ov.save_model(ov_model, "{}_combined.xml".format(model_path.stem))

if __name__ == "__main__":
    main()