import cv2
import os
import argparse
import glob
import numpy as np
import torch
from torch.autograd import Variable
from utils import *
from networks import *
import nvidia_smi
import time

parser = argparse.ArgumentParser(description="PReNet_Test")
parser.add_argument("--input", type=str, default="/home/ubuntu/code/datasets/test/Rain100L/rainy_image/", help='path to training data')
parser.add_argument("--output_path", type=str, default="/home/ubuntu/code/results_detect/Rain100L/", help='path to save results')
parser.add_argument("--logdir", type=str, default="logs/real/", help='path to model and log files')
parser.add_argument("--which_model", type=str, default="PReNet.pth", help='model name')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
opt = parser.parse_args()

nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)


def derain(img_name, output_path):
    
    if opt.use_GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

    total_time = time.time()

    os.makedirs(output_path, exist_ok=True)

    load_model_time = time.time()

    # Build model
    print('Loading model ...\n')
    model = PReNet(opt.recurrent_iter, opt.use_GPU)
    print_network(model)
    if opt.use_GPU:
        model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, opt.which_model)))
    model.eval()

    load_model_time = time.time() - load_model_time
    print('load model time:', load_model_time)

    process_time = 0
    read_time = 0
    write_time = 0

    read_time = time.time()


    # input image
    y = cv2.imread(img_name)
    b, g, r = cv2.split(y)
    y = cv2.merge([r, g, b])
    #y = cv2.resize(y, (int(500), int(500)), interpolation=cv2.INTER_CUBIC)

    y = normalize(np.float32(y))
    y = np.expand_dims(y.transpose(2, 0, 1), 0)
    y = Variable(torch.Tensor(y), requires_grad=True)

    read_time = time.time() - read_time

    if opt.use_GPU:
        y = y.cuda()

    with torch.no_grad():
        if opt.use_GPU:
            torch.cuda.synchronize()
        process_time = time.time()

        out, _ = model(y)
        out = torch.clamp(out, 0., 1.)

        if opt.use_GPU:
            torch.cuda.synchronize()
        process_time = time.time() - process_time

    write_time = time.time()

    if opt.use_GPU:
        save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())   #back to cpu
    else:
        save_out = np.uint8(255 * out.data.numpy().squeeze())

    save_out = save_out.transpose(1, 2, 0)
    b, g, r = cv2.split(save_out)
    save_out = cv2.merge([r, g, b])

    derain_path = os.path.join(output_path, img_name.split("/")[-1])

    cv2.imwrite(derain_path, save_out)

    write_time = time.time() - write_time
    total_time = time.time() - total_time

    return load_model_time, read_time, process_time, write_time, total_time

if __name__ == "__main__":
    output_path = opt.output_path
    input = opt.input

    load_model_time, read_time, process_time, write_time, total_time = derain(input, output_path)
    print('load derain model time:', load_model_time)
    print('read time:', read_time)
    print('process time:', process_time)
    print('write time:', write_time)
    print('total derain time: ', total_time)
    print('The path of derain result image is ',os.path.join(output_path, input.split("/")[-1]) )
    print('---------------------')
