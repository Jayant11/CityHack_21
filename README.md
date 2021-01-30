# SmartVision

SmartVision is a social distancing detector implemented in Python with OpenCV and Tensorflow.<br/>
Take a peek at the backend program in work 😊

![](/img/result.gif)

# Installation - Linux

### OpenCV
```bash
pip install opencv-contrib-python
```

### Other requirements
All the other requirements can be installed via the command : 
```bash
pip install -r requirements.txt
```

# Download Tensorflow models

Download the faster_rcnn_inception_resnet_v2_atrous_coco model from Tensorflow by clicking [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md).
Just download all the models you want to try out, put them in the models folder and unzip them. For example :
```bash
tar -xvzf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
```

# Run project

### Calibrate
Run 
```bash
python calibrate_with_mouse.py
```
You will be asked as input the name of the video and the size of the frame you want to work with.
**Note** : It is important to start with the top right corner, than the bottom right, then bottom left, than end by top left corner !
You can add any video to the video folder and work with that.

### Start social distancing detection
Run 
```bash
python social_distanciation_video_detection.py
```
You will be asked as inputs :
- The tensorflow model you want to use. Enter digit from shown options.
- The name of the video. Enter digit from shown options.
- The distance (in pixels between 2 persons).

# Outputs
Both video outputs (normal frame and bird eye view) will be stored in the outputs file.
