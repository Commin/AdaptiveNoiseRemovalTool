import numpy as np
import time
import csv
import os
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

parser = argparse.ArgumentParser(description="classification")
parser.add_argument("--data_path", type=str, default="/home/ubuntu/code/datasets/test/Rain100L/rainy_image/", help='path to training data')
parser.add_argument("--model_path", type=str, default="../vgg19_drop_batch_best_weights_256.h5", help='path to training data')
parser.add_argument("--save_path", type=str, default="/home/ubuntu/code/results_detect/Rain100L/", help='path to save results')
opt = parser.parse_args()


def label_to_weather(label):

    if label == 0:
        return 'cloudy'
    elif label == 1:
        return 'foggy'
    elif label == 2:
        return 'rainy'
    elif label == 3:
        return 'shine'
    else:
        return 'sunrise'


if __name__ == '__main__':

    #Define Path
    #model_path = '../CNN_best_weights_256.h5'
    #model_path = '../CNN_augmentation_best_weights_256.h5'
    #model_path = '../vgg16_best_weights_256.h5'
    #model_path = '../vgg16_aug_best_weights_256.h5'
    #model_path = '../vgg16_drop_batch_best_weights_256.h5'
    #model_path = '../vgg19_drop_batch_best_weights_256.h5'
    #model_path = '../resnet101_drop_batch_best_weights_256.h5'

    headers = ['image', 'label', 'weather']
    rows = []

    model_path = opt.model_path
    save_path = opt.save_path

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
        item = {"image":img_names[i], "label": str(image_labels[i]) , "weather": label_to_weather(image_labels[i])}
        rows.append(item)

    with open( save_path +'/label_result.csv','w') as f:
        f_csv = csv.DictWriter(f, headers)
        f_csv.writeheader()
        f_csv.writerows(rows)

    print('wrote data into csv file.')

    print('weather classification completed.')
    print('---------------------')


