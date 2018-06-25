#! /usr/bin/env python

import os
import argparse
import json
import cv2
from background_subtract import background_sub
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm
import numpy as np
import nltk 


# text = "Demonhunter is standing on the badland on the right of sunwell."
text = "Demonhunter is running on the marbleland from the right of sunwell."
character_width = 190
character_height = 190
character_start_width = [] 
character_start_height = [] 
object_label = ""
object_ymin = [] 
object_ymax = [] 
object_xmin = [] 
object_xmax = [] 
character_detection = []
bound = 40
images_background = []
background_path = "dataVideo/badland.mp4"
character_path = ""


speed = 0
speed_increase = 0
obj_direction = "left"
def get_object_img(boxes,image):
    if(len(boxes)>0):
        box = boxes[0]
        cropImg = image[box.ymin:box.ymax,box.xmin:box.xmax]
        if cropImg.shape[0]>0 and cropImg.shape[1] > 0:
            cropImg = cv2.resize(cropImg,(character_width,character_height),interpolation=cv2.INTER_CUBIC)
            character_detection.append(cropImg.copy())
            cropImg = background_sub(cropImg)
            return cropImg
    # image1 = image.copy()
    # cropImg = cv2.resize(image1,(character_width,character_height),interpolation=cv2.INTER_CUBIC)
    # cropImg = cv.CreateImage((character_width,character_height),8,3)
    cropImg = np.zeros((character_width,character_height,3), np.uint8)
    # cropImg = np.zeros([character_width,character_height,3])
    # cropImg = 0
    # image1[0:box.ymin,:] = 0
    # image1[box.ymax:,:] = 0
    # image1[:,0:box.xmin] = 0
    # image1[:,box.xmax:] = 0
    return cropImg

def cordinate_optimization():
    difference = 20000 
    for i in range(len(object_xmax)):
        if i > 0 :
            if (object_xmax[i] - object_xmax[i-1])*(object_xmax[i] - object_xmax[i-1]) < difference:
                object_xmax[i] = object_xmax[i-1]
            if (object_ymax[i] - object_ymax[i-1])*(object_ymax[i] - object_ymax[i-1]) < difference:
                object_ymax[i] = object_ymax[i-1]
            if (object_ymin[i] - object_ymin[i-1])*(object_ymin[i] - object_ymin[i-1]) < difference:
                object_ymin[i] = object_ymin[i-1]

def build_animation(character):
    cordinate_optimization()
    character_start_x = object_xmax 
    if obj_direction == "left":
        character_start_y = object_ymin  
    if obj_direction == "right":
        character_start_y = object_ymax  
    # print(character_start_y)
    # print(character_start_x)
    # print(object_ymax[0])
    # print(object_xmax[0])
    background_reader = cv2.VideoCapture(background_path)

    background_frames = int(background_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    background_h = int(background_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    background_w = int(background_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_writer = cv2.VideoWriter("outputVideo/output.mp4",
                            cv2.VideoWriter_fourcc(*'MPEG'), 
                            50.0, 
                            (background_w, background_h))
                            # (150, 150))

    video_writer2 = cv2.VideoWriter("outputVideo/character_subtraction.mp4",
                            cv2.VideoWriter_fourcc(*'MPEG'), 
                            50.0, 
                            (character_width, character_height))
    video_writer3 = cv2.VideoWriter("outputVideo/character_detection.mp4",
                            cv2.VideoWriter_fourcc(*'MPEG'), 
                            50.0, 
                            (character_width, character_height))



    # the main loop
    show_window = False
    previous_image = 0
    length = len(character) 
    index = 0
    for i in character_detection:
        video_writer3.write(i)
    # fill frame
    while(len(character)<len(images_background)):
        if(np.array_equal(np.zeros((character_width,character_height,3), np.uint8),character[index])):
            character[index] = previous_image
        else:
            previous_image = character[index]
        if index < length:
            character.append(character[index])
            index = index + 1
        else:
            index = 0
    speed = 0
    for i in range(len(images_background)):
        video_writer3.write(character[i]) 
        for w in range(character[i].shape[0]):
            for h in range(character[i].shape[1]):
                if(not (character[i][w,h][0] < bound and character[i][w,h][1] < bound and character[i][w,h][2] < bound )):
                    images_background[i][character_start_y[i] +w, character_start_x[i]+speed+h] = character[i][w,h]
        video_writer.write(images_background[i]) 
        video_writer2.write(character[i]) 
        if(character_start_x[i]+speed+character_width < background_w):
            speed = speed + speed_increase
    if show_window: cv2.destroyAllWindows()
    video_writer.release()       
    video_writer2.release()       
    video_writer3.release()       

def _main_(args):
    # config_path  = args.conf
    config_path  = "config.json"
    input_path   = character_path 
    # output_path  = args.output

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    # makedirs(output_path)

    ###############################
    #   Set some parameter
    ###############################       
    net_h, net_w = 416, 416 # a multiple of 32, the smaller the faster
    obj_thresh, nms_thresh = 0.5, 0.45

    ###############################
    #   Load the model
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    infer_model = load_model(config['train']['saved_weights_name'])

    ###############################
    #   Predict bounding boxes 
    ###############################
    if input_path[-4:] == '.mp4': # do detection on a video  
        # video_out = output_path + input_path.split('/')[-1]
        video_out = "outputVideo/Fullvideo_detection.mp4"
        video_reader = cv2.VideoCapture(input_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'MPEG'), 
                               50.0, 
                               (frame_w, frame_h))

        background_reader = cv2.VideoCapture(background_path)

        background_frames = int(background_reader.get(cv2.CAP_PROP_FRAME_COUNT))

        # the main loop
        batch_size  = 1
        images      = []
        character_image = []
        start_point = 0 #%
        show_window = False
        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()

            if (float(i+1)/nb_frames) > start_point/100.:
                images += [image]

                if (i%batch_size == 0) or (i == (nb_frames-1) and len(images) > 0):
                    # predict the bounding boxes
                    try:
                        batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)

                        for i in range(len(images)):
                            character = get_object_img(batch_boxes[i],images[i])
                            character_image.append(character)
                            # draw bounding boxes on the image using labels
                            draw_boxes(images[i], batch_boxes[i], config['model']['labels'], obj_thresh)   

                            # show the video with detection bounding boxes          
                            if show_window: cv2.imshow('video with bboxes', images[i])  

                            # write result to the output video
                            video_writer.write(images[i]) 
                        images = []
                    except AttributeError:
                        print("not found shape")
                if show_window and cv2.waitKey(1) == 27: break  # esc to quit

        # Background detection 
        print("background detection---------------------------------------------")
        batch_size  = 1
        images      = []
        start_point = 0 #%
        show_window = False
        for i in tqdm(range(background_frames)):
            _, image = background_reader.read()

            if (float(i+1)/background_frames) > start_point/100.:
                images += [image]

                if (i%batch_size == 0) or (i == (background_frames-1) and len(images) > 0):
                    # predict the bounding boxes
                    try:
                        batch_boxes = get_yolo_boxes(infer_model, images, net_h, net_w, config['model']['anchors'], obj_thresh, nms_thresh)
                        for i in range(len(images)):
                            if(len(batch_boxes[i])>0):
                                images_background.append(images[i])
                                box = batch_boxes[i][0]
                                object_xmax.append(box.xmax)
                                object_ymax.append(box.ymax)
                                object_xmin.append(box.xmin)
                                object_ymin.append(box.ymin)
                        images = []
                    except AttributeError:
                        print("not found shape")
                if show_window and cv2.waitKey(1) == 27: break  # esc to quit
        print("make animation----------------------")
        build_animation(character_image)
        print("done----------------------")
        if show_window: cv2.destroyAllWindows()
        video_reader.release()
        video_writer.release()       

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Predict with a trained yolo model')
    # # argparser.add_argument('-c', '--conf', help='path to configuration file')
    # argparser.add_argument('-s', '--str', help='input text')
    # # argparser.add_argument('-i', '--input', help='path to an image, a directory of images, a video, or webcam')    
    # # argparser.add_argument('-o', '--output', default='output/', help='path to output directory')   
    word = nltk.word_tokenize(text)
    word_tag = nltk.pos_tag(word)
    character = []
    movement = []
    place = []
    obj = []
    direction = []
    current_place = ["snowland","badland","marbleland"]
    current_character = ["Demonhunter","Blademaster"]
    current_movement = ["stand","run"]
    current_object = ["sunwell"]
    current_direction = ["right","left"]
    stemmer = nltk.PorterStemmer()
    for i in word_tag:
        if i[1] == "NNP":
            character.append(i[0])
        if i[1] == "VBG" or i[1] == "VB":
            movement.append(stemmer.stem(i[0]))
        if i[1] == "NN":
            if i[0] in current_place:
                place.append(i[0])
            if i[0] in current_object:
                obj.append(i[0])
            if i[0] in current_direction:
                direction.append(i[0])

    character_path = "dataVideo/"+character[0] +"_"+movement[0]+".mp4"
    background_path = "dataVideo/" + place[0]+".mp4"
    print(character_path)
    print(background_path)
    if movement[0] == "run":
        speed_increase = 1
    if movement[0] == "stand":
        speed_increase = 0
    obj_direction = direction[0]
    args = argparser.parse_args()
    _main_(args)
