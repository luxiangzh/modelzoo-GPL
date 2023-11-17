import os
import argparse
import torch
import torch_aie
from torch_aie import _enums


def aie_compile(traced_model, args):
    input_shape = (args.batch_size, 3, args.img_size, args.img_size)

    traced_model.eval()
    print("torch_aie compile start !")
    torch_aie.set_device(0)
    compile_inputs = [ torch_aie.Input(shape = input_shape, dtype = torch.float16, format = torch_aie.TensorFormat.NCHW) ]
    compiled_model = torch_aie.compile(
        traced_model,
        inputs = compile_inputs,
        precision_policy = _enums.PrecisionPolicy.FP16,
        soc_version = "Ascend310P3",
        optimization_level = 0
    )
    print("torch_aie compile done !")
    print("compiled model is ", compiled_model.graph)

    compiled_model.save(args.pt_dir)
    print("torch aie compiled model saved. ")


if __name__ == '__main__':   
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--ts_model",
                        default="./yolov5s_v6.torchscript.pt",
                        type=str,
                        help="The original torch pt file from pretraining")   
    parser.add_argument("--save_dir",
                        default="./",
                        type=str,
                        help="The path of the directory that stores the compiled model")   
    parser.add_argument('--batch_size',
                        default=8,
                        type=int,
                        help="batch size")
    parser.add_argument('--img_size', 
                        type=int, 
                        default=640, 
                        help='image size')
    args = parser.parse_args()

    traced_model = torch.jit.load(args.ts_model)
    traced_model.eval()
    base_yolo_name = os.path.basename(args.ts_model).split('.')[0] + "_bs{}_aie.pt".format(args.batch_size)
    args.pt_dir = os.path.join(args.save_dir, base_yolo_name)
    
    aie_compile(traced_model, args)
    