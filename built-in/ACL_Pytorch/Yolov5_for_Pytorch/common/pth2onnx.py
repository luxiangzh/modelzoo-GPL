# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
from collections import OrderedDict

import torch
import onnx
import onnxsim

from mobilenetv3 import MobileNetV3_Small


def update_model(checkpoint):
    new_state_dict = OrderedDict()
    for key, value in checkpoint.items():
        if key == "module.features.0.0.weight":
            print(value)
        if key[0:7] == "module.":
            name = key[7:]
        else:
            name = key[0:]

        new_state_dict[name] = value
    return new_state_dict


def load_model(args):
    checkpoint = torch.load(args.pth_model, map_location='cpu')['state_dict']
    checkpoint = update_model(checkpoint)
    model_pt = MobileNetV3_Small()
    model_pt.load_state_dict(checkpoint)
    model_pt.eval()
    return model_pt


def pth2onnx(args, model_pt):
    input_names = ["input"]
    output_names = ["output"]
    input_data = torch.randn(args.batch_size, 3, 224, 224)
    output_onnx = os.path.join(args.output_dir, args.onnx_model)
    torch.onnx.export(
        model_pt, input_data, output_onnx,
        input_names=input_names, output_names=output_names,
        opset_version=args.opset, export_params=True, verbose=False, do_constant_folding=True,
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}} if args.dynamic else None)

    # Checks
    model_onnx = onnx.load(output_onnx)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Simplify
    if args.simplify:
        try:
            print(f'simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                test_input_shapes={'input': list(input_data.shape)} if args.dynamic else None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, output_onnx)
        except Exception as e:
            print(f'simplifier failure: {e}')

    return model_onnx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="export MobileNetV3 onnx")
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--pth-model', type=str, default='model.pt')
    parser.add_argument('--onnx-model', type=str, default='model.onnx')
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--opset', type=int, default=11, help='ONNX: opset version')
    parser.add_argument('--dynamic', action='store_true', help='ONNX: dynamic axes')
    parser.add_argument('--simplify', action='store_true', help='ONNX: simplify model')
    args = parser.parse_args()

    ### load model
    model_pt = load_model(args)

    ### pth2onnx
    model_onnx = pth2onnx(args, model_pt)
    print("导出onnx success")


# 2.0/3.0
model = torch.load(opt.weights, map_location=torch.device('cpu'))['model'].float()
model.eval()
model.model[-1].export = True  # set Detect() layer export=True
import onnx

print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
f = opt.weights.replace('.pt', '.onnx')  # filename
model.fuse()  # only for ONNX
torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                  output_names=['classes', 'boxes'] if y is None else ['output'])

# Checks
onnx_model = onnx.load(f)  # load onnx model
onnx.checker.check_model(onnx_model)  # check onnx model
print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
print('ONNX export success, saved as %s' % f)


# 3.1/4.0/5.0
model = attempt_load(opt.weights, map_location=torch.device('cpu'))  # load FP32 model

# Update model
for k, m in model.named_modules():
    m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
    if isinstance(m, models.common.Conv):  # assign export-friendly activations
        if isinstance(m.act, nn.Hardswish):
            m.act = Hardswish()
        elif isinstance(m.act, nn.SiLU):
            m.act = SiLU()
    # elif isinstance(m, models.yolo.Detect):
    #     m.forward = m.forward_export  # assign forward (optional)

model.model[-1].export = True  # set Detect() layer export=True

try:
    import onnx

    print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
    f = opt.weights.replace('.pt', '.onnx')  # filename
    torch.onnx.export(model, img, f, verbose=False, opset_version=12, input_names=['images'],
                      output_names=['classes', 'boxes'] if y is None else ['output'])

    # Checks
    onnx_model = onnx.load(f)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
    print('ONNX export success, saved as %s' % f)
except Exception as e:
    print('ONNX export failure: %s' % e)