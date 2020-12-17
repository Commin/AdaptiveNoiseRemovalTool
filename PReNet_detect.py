import cv2
import os
import argparse
import glob
import numpy as np
import torch
from torch.autograd import Variable
from utils import *
from networks import *
import detect
import time
import csv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

parser = argparse.ArgumentParser(description="Detect_Test")
parser.add_argument("--logdir", type=str, default="logs/PReNet6/", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="/home/ubuntu/code/datasets/test/Rain100L/rainy_image/", help='path to training data')
parser.add_argument("--save_path", type=str, default="/home/ubuntu/code/results_detect/Rain100L/", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
parser.add_argument('--weights', help='Path to model weights', type=str, default='/home/ubuntu/code/yolov3-detect/yolov3.weights')
parser.add_argument('--configs', help='Path to model configs',type=str, default='/home/ubuntu/code/yolov3-detect/yolov3.cfg')
parser.add_argument('--class_names', help='Path to class-names text file', type=str, default='/home/ubuntu/code/yolov3-detect/coco.names')
parser.add_argument('--conf_thresh', help='Confidence threshold value', default=0.5)
parser.add_argument('--nms_thresh', help='Confidence threshold value', default=0.4)
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

def derain(paths):
    
    derain_paths = []
    #derain_images = []
    

    total_start_time = time.time()

    os.makedirs(opt.save_path, exist_ok=True)

    load_model_time = time.time()

    # Build model
    print('Loading model ...\n')
    model = PReNet(opt.recurrent_iter, opt.use_GPU)
    print_network(model)
    if opt.use_GPU:
        model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_latest.pth')))
    model.eval()

    load_model_time = time.time() - load_model_time
    print('load model time:', load_model_time)

    total_process_time = 0
    total_read_time = 0
    total_write_time = 0
    count = 0

    headers = ['image', 'read_time', 'process_time', 'write_time']
    rows = []

    for img_path in paths:
        if is_image(img_name):

            read_start_time = time.time()

            img_name = img_path.split("/")[-1]

            # input image
            y = cv2.imread(img_path)
            b, g, r = cv2.split(y)
            y = cv2.merge([r, g, b])
            #y = cv2.resize(y, (int(500), int(500)), interpolation=cv2.INTER_CUBIC)

            y = normalize(np.float32(y))
            y = np.expand_dims(y.transpose(2, 0, 1), 0)
            y = Variable(torch.Tensor(y))

            read_end_time = time.time()
            read_dur_time = read_end_time - read_start_time
            total_read_time += read_dur_time

            if opt.use_GPU:
                y = y.cuda()

            with torch.no_grad(): #
                if opt.use_GPU:
                    torch.cuda.synchronize()
                process_start_time = time.time()

                out, _ = model(y)
                out = torch.clamp(out, 0., 1.)

                if opt.use_GPU:
                    torch.cuda.synchronize()
                process_end_time = time.time()
                process_dur_time = process_end_time - process_start_time
                total_process_time += process_dur_time
    
                print(img_name, ': ', process_dur_time)
            
            write_start_time = time.time()

            if opt.use_GPU:
                save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())   #back to cpu
            else:
                save_out = np.uint8(255 * out.data.numpy().squeeze())

            save_out = save_out.transpose(1, 2, 0)
            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])

            #derain_images.append(save_out)
            derain_path = os.path.join(opt.save_path, img_name)
            derain_paths.append(derain_path)

            cv2.imwrite(derain_path, save_out)

            write_end_time = time.time()
            write_dur_time = write_end_time - write_start_time
            total_write_time += write_dur_time

            item = {"image":img_name,"read_time":read_dur_time, "process_time":process_dur_time, "write_time": write_dur_time }
            rows.append(item)
            count += 1

    avg_read_time = total_read_time/count
    avg_process_time = total_process_time/count
    avg_write_time = total_write_time/count
    total_time = time.time() - total_start_time

    print('average read time:', avg_read_time)
    print('average process time:', avg_process_time)
    print('average write time:', avg_write_time)
    print('load model time:', load_model_time)
    print('total time: ', total_time)

    rows.append({"image": "average" , "read_time":avg_read_time, "process_time":avg_process_time, "write_time": avg_write_time})
    rows.append({"image": "load_model_time", "read_time" : load_model_time})
    rows.append({"image": "total_time", "read_time" : total_time})

    with open( opt.save_path +'/time_result.csv','w') as f:
        f_csv = csv.DictWriter(f, headers)
        f_csv.writeheader()
        f_csv.writerows(rows)

    print('wrote data into csv file.')

    return derain_paths

def classify():
    #Define Path
    #model_path = '../CNN_best_weights_256.h5'
    #model_path = '../CNN_augmentation_best_weights_256.h5'
    #model_path = '../vgg16_best_weights_256.h5'
    #model_path = '../vgg16_aug_best_weights_256.h5'
    #model_path = '../vgg16_drop_batch_best_weights_256.h5'
    model_path = '../vgg19_drop_batch_best_weights_256.h5'
    #model_path = '../resnet101_drop_batch_best_weights_256.h5'

    #Load the pre-trained models

    load_model_time = time.time()

    model = load_model(model_path)
    load_model_time = time.time() - load_model_time
    print('load model time:', load_model_time)

    img_width, img_height = 256, 256
    folder_path = opt.data_path 

    # load all images into a list
    cropped_images = []
    image_labels = []
    img_names = []

    total_crop_time = 0
    total_classify_time = 0

    total_crop_time = time.time()

    for img in os.listdir(folder_path):
        img_name = os.path.join(folder_path, img)
        img = image.load_img(img_name, target_size=(img_width, img_height))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        cropped_images.append(img)
        img_names.append(img_name)
        
    print('loaded '+ str(len(cropped_images)) + ' images')

    total_crop_time = time.time() - total_crop_time
    print('average read image time:', total_crop_time/len(img_names))

    total_classify_time = time.time()

    # stack up images list to pass for prediction
    test_images = np.vstack(cropped_images)
    y_pred = model.predict(test_images, batch_size=1, verbose=1)
    classes = np.argmax(y_pred, axis = 1)
    image_labels = classes.tolist()
    print(image_labels)

    total_classify_time = time.time() - total_classify_time
    print('average classify time:', total_classify_time/len(img_names))

    derain_path = []
    for i in range(0,len(img_names)):
        if image_labels[i] == 2:
            print(img_names[i])
            derain_path.append(img_names[i])

    return img_names, image_labels, derain_path


if __name__ == "__main__":

    img_names, image_labels, derain_path = classify()

    print('weather classification completed.')
    print('---------------------')

    # return derain paths
    derain_paths = derain(derain_path)
    
    print('deraining completed.')
    print('--------------------')

    detect_paths = []
    for i in len(img_names):
        img_name = os.path.join(folder_path, img)
        detect_paths.append(img_name)
    
    detect.detect_image_path(detect_paths, output_path, yolo_weights, yolo_cfg, coco_names, confidence_threshold, nms_threshold)
    print("Done All")

    
