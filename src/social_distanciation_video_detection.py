from bird_view_transfo_functions import compute_perspective_transform,compute_point_perspective_transformation
from tf_model_object_detection import Model 
from colors import bcolors
import numpy as np
import itertools
import imutils
import time
import math
import glob
import yaml
import cv2
import os

COLOR_RED = (0, 0, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (255, 0, 0)
BIG_CIRCLE = 60
SMALL_CIRCLE = 3


def get_human_box_detection(boxes,scores,classes,height,width):
	array_boxes = list() # Create an empty list
	for i in range(boxes.shape[1]):
		# If the class of the detected object is 1 and the confidence of the prediction is > 0.6
		if int(classes[i]) == 1 and scores[i] > 0.75:
			box = [boxes[0,i,0],boxes[0,i,1],boxes[0,i,2],boxes[0,i,3]] * np.array([height, width, height, width])
			# Add the results converted to int
			array_boxes.append((int(box[0]),int(box[1]),int(box[2]),int(box[3])))
	return array_boxes


def get_centroids_and_groundpoints(array_boxes_detected):
	array_centroids,array_groundpoints = [],[] # Initialize empty centroid and ground point lists 
	for index,box in enumerate(array_boxes_detected):
		centroid,ground_point = get_points_from_box(box)
		array_centroids.append(centroid)
		array_groundpoints.append(centroid)
	return array_centroids,array_groundpoints


def get_points_from_box(box):
	center_x = int(((box[1]+box[3])/2))
	center_y = int(((box[0]+box[2])/2))
	center_y_ground = center_y + ((box[2] - box[0])/2)
	return (center_x,center_y),(center_x,int(center_y_ground))


def change_color_on_topview(pair):
	cv2.circle(bird_view_img, (pair[0][0],pair[0][1]), BIG_CIRCLE, COLOR_RED, 2)
	cv2.circle(bird_view_img, (pair[0][0],pair[0][1]), SMALL_CIRCLE, COLOR_RED, -1)
	cv2.circle(bird_view_img, (pair[1][0],pair[1][1]), BIG_CIRCLE, COLOR_RED, 2)
	cv2.circle(bird_view_img, (pair[1][0],pair[1][1]), SMALL_CIRCLE, COLOR_RED, -1)

def draw_rectangle(corner_points):
	# Draw rectangle box over the delimitation area
	cv2.line(frame, (corner_points[0][0], corner_points[0][1]), (corner_points[1][0], corner_points[1][1]), COLOR_BLUE, thickness=1)
	cv2.line(frame, (corner_points[1][0], corner_points[1][1]), (corner_points[3][0], corner_points[3][1]), COLOR_BLUE, thickness=1)
	cv2.line(frame, (corner_points[0][0], corner_points[0][1]), (corner_points[2][0], corner_points[2][1]), COLOR_BLUE, thickness=1)
	cv2.line(frame, (corner_points[3][0], corner_points[3][1]), (corner_points[2][0], corner_points[2][1]), COLOR_BLUE, thickness=1)


print(bcolors.WARNING +"[ Loading config file for the bird view transformation ] "+ bcolors.ENDC)
with open("../conf/config_birdview.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile)
width_og, height_og = 0,0
corner_points = []
for section in cfg:
	corner_points.append(cfg["image_parameters"]["p1"])
	corner_points.append(cfg["image_parameters"]["p2"])
	corner_points.append(cfg["image_parameters"]["p3"])
	corner_points.append(cfg["image_parameters"]["p4"])
	width_og = int(cfg["image_parameters"]["width_og"])
	height_og = int(cfg["image_parameters"]["height_og"])
	img_path = cfg["image_parameters"]["img_path"]
	size_frame = cfg["image_parameters"]["size_frame"]
print(bcolors.OKGREEN +" Done : [ Config file loaded ] ..."+bcolors.ENDC )


model_names_list = [name for name in os.listdir("../models/.") if name.find(".") == -1]
for index,model_name in enumerate(model_names_list):
    print(" - {} [{}]".format(model_name,index))
model_num = input(" Please select the number related to the model that you want : ")
if model_num == "":
	model_path="../models/faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb" 
else :
	model_path = "../models/"+model_names_list[int(model_num)]+"/frozen_inference_graph.pb"
print(bcolors.WARNING + " [ Loading the TENSORFLOW MODEL ... ]"+bcolors.ENDC)
model = Model(model_path)
print(bcolors.OKGREEN +"Done : [ Model loaded and initialized ] ..."+bcolors.ENDC)


video_names_list = [name for name in os.listdir("../video/") if name.endswith(".mp4") or name.endswith(".avi")]
for index,video_name in enumerate(video_names_list):
    print(" - {} [{}]".format(video_name,index))
video_num = input("Enter the exact name of the video (including .mp4 or else) : ")
if video_num == "":
	video_path="../video/PETS2009.avi"  
else :
	video_path = "../video/"+video_names_list[int(video_num)]


distance_minimum = input("Prompt the size of the minimal distance between 2 pedestrians : ")
if distance_minimum == "":
	distance_minimum = "110"


matrix,imgOutput = compute_perspective_transform(corner_points,width_og,height_og,cv2.imread(img_path))
height,width,_ = imgOutput.shape
blank_image = np.zeros((height,width,3), np.uint8)
height = blank_image.shape[0]
width = blank_image.shape[1] 
dim = (width, height)


vs = cv2.VideoCapture(video_path)
output_video_1,output_video_2 = None,None
# Loop until the end of the video stream
while True:	
	# Load the image of the ground and resize it to the correct size
	img = cv2.imread("../img/chemin_1.png")
	bird_view_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	
	# Load the frame
	(frame_exists, frame) = vs.read()
	# Test if it has reached the end of the video
	if not frame_exists:
		break
	else:
		# Resize the image to the correct size
		frame = imutils.resize(frame, width=int(size_frame))
		
		(boxes, scores, classes) =  model.predict(frame)

		array_boxes_detected = get_human_box_detection(boxes,scores[0].tolist(),classes[0].tolist(),frame.shape[0],frame.shape[1])
		
		array_centroids,array_groundpoints = get_centroids_and_groundpoints(array_boxes_detected)

		transformed_downoids = compute_point_perspective_transformation(matrix,array_groundpoints)
		
		for point in transformed_downoids:
			x,y = point
			cv2.circle(bird_view_img, (x,y), BIG_CIRCLE, COLOR_GREEN, 2)
			cv2.circle(bird_view_img, (x,y), SMALL_CIRCLE, COLOR_GREEN, -1)

		if len(transformed_downoids) >= 2:
			for index,downoid in enumerate(transformed_downoids):
				if not (downoid[0] > width or downoid[0] < 0 or downoid[1] > height+200 or downoid[1] < 0 ):
					cv2.rectangle(frame,(array_boxes_detected[index][1],array_boxes_detected[index][0]),(array_boxes_detected[index][3],array_boxes_detected[index][2]),COLOR_GREEN,2)

			list_indexes = list(itertools.combinations(range(len(transformed_downoids)), 2))
			for i,pair in enumerate(itertools.combinations(transformed_downoids, r=2)):
				if math.sqrt( (pair[0][0] - pair[1][0])**2 + (pair[0][1] - pair[1][1])**2 ) < int(distance_minimum):
					if not (pair[0][0] > width or pair[0][0] < 0 or pair[0][1] > height+200  or pair[0][1] < 0 or pair[1][0] > width or pair[1][0] < 0 or pair[1][1] > height+200  or pair[1][1] < 0):
						change_color_on_topview(pair)
						index_pt1 = list_indexes[i][0]
						index_pt2 = list_indexes[i][1]
						cv2.rectangle(frame,(array_boxes_detected[index_pt1][1],array_boxes_detected[index_pt1][0]),(array_boxes_detected[index_pt1][3],array_boxes_detected[index_pt1][2]),COLOR_RED,2)
						cv2.rectangle(frame,(array_boxes_detected[index_pt2][1],array_boxes_detected[index_pt2][0]),(array_boxes_detected[index_pt2][3],array_boxes_detected[index_pt2][2]),COLOR_RED,2)

	draw_rectangle(corner_points)
	# Show both images	
	cv2.imshow("Bird view", bird_view_img)
	cv2.imshow("Original picture", frame)


	key = cv2.waitKey(1) & 0xFF

	# Write the both outputs video to a local folders
	if output_video_1 is None and output_video_2 is None:
		fourcc1 = cv2.VideoWriter_fourcc(*"MJPG")
		output_video_1 = cv2.VideoWriter("../output/video.avi", fourcc1, 25,(frame.shape[1], frame.shape[0]), True)
		fourcc2 = cv2.VideoWriter_fourcc(*"MJPG")
		output_video_2 = cv2.VideoWriter("../output/bird_view.avi", fourcc2, 25,(bird_view_img.shape[1], bird_view_img.shape[0]), True)
	elif output_video_1 is not None and output_video_2 is not None:
		output_video_1.write(frame)
		output_video_2.write(bird_view_img)

	# Break the loop
	if key == ord("q"):
		break
