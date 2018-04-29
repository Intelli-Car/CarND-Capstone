This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).


[//]: # (Image References)
[image0]: ./readme_images/final-project.png "System Architecture Diagram"
[image1]: ./readme_images/dataset_folder_structure.jpg "Dataset Folder Structure"
[image2]: ./readme_images/labelImg_real_image.jpg "Label Real Image"
[image3]: ./readme_images/labelImg_sim_image.jpg "Label Sim Image"
[image4]: ./readme_images/labelImg_xml_real_image.jpg "XML Real Image"
[image5]: ./readme_images/object_distribution.jpg "Object Distribution"
[image6]: ./readme_images/similar_image_series.jpg "Similar Image Series"
[image7]: ./readme_images/loss_graph_ssd_mobilenet_v1_coco_on_sim_data.jpg "Loss Graph SSD MobilenetV1 On Sim Data"
[image8]: ./readme_images/result_ssd_mobilenet_v1_coco_on_sim_data.jpg "Result SSD MobilenetV1 on Sim Data"
[image9]: ./readme_images/loss_graph_ssd_mobilenet_v1_coco_on_real_data.jpg "Loss Graph SSD MobilenetV1 On Real Data"
[image10]: ./readme_images/traffic_node.png "Traffic Node"
[image11]: ./readme_images/waypoint_updater.png "Waypoint Updater Node"
[image12]: ./readme_images/dbw_node.png "DBW Node"
[image13]: ./readme_images/autoware_node.png "Autoware Node"

### The team "Intelli-car"

|              |     Name         | Email | Timezone | Slack |
|--------------|------------------|----------|----------|--------------------------------|
|              | Qingqing Xia     | qingqing.xia.2015@gmail.com | @qingqing | UTC+1 (Germany) |
|              | Ananthesh J Shet | shetanantheshnlsv@yahoo.in  |  @anantheshjshet | UTC+05:30(India) |
|              | Andre Marais     | agmarais@gmail.com |  @veldrin | UTC + 2 (South Africa) |
|              | HanByul Yang     | hanbyul.yang@gmail.com | @hb | UTC+09:00 (South Korea) |
| Team lead    | Hasan Chowdhury  | shemonc@gmail.com | @shemon | UTC-5 (Ottawa) |

### 1. Submission checklist and requirements

- Launch correctly using the launch files provided in the capstone repo. The launch/styx.launch and launch/site.launch are used to test code in the simulator and on the vehicle respectively. The submission size limit is within 2GB.  
- Car smoothly follows waypoints in the simulator.  
- Car follows the target top speed set for the waypoints' twist.twist.linear.x in waypoint_loader.py. This has been tested by setting different velocity parameter(Kph) in /ros/src/waypoint_loader/launch/waypoint_loader.launch.  
- Car stops at traffic lights when needed.  
- Depending on the state of /vehicle/dbw_enabled, Car stop and restart PID controllers  
- Publish throttle, steering, and brake commands at 50hz.  
- Finally for Simulated Track 1 we tested with velocity 40 Kmph and for track 2, 10 Kmph as the default velocity given in corresponding launch file.[I will double check this and remove this comment after that]

### 2. System Architecture Diagram

Following diagram describes the overall system archicture showing ROS nodes and topics those are used to communicate among different parts of this 'Self Driving' vehicle susbsystems for this project
![alt text][image0]

#### 2.1 Traffic Light detection Node
This node takes in data from the /image_color, /current_pose, and /base_waypoints topics and publishes the locations to stop for red traffic lights to the /traffic_waypoint topic.

![alt text][image10]

Here we introduce a new ROS message named Light (int32 index, uint8 sate) to publish not only the index of the closest traffic light but also the state(COLOR) of it and publish into node /traffic_waypoint. Waypoint updater node (see bellow) will use this information
to slow down for a closest Yellow or RED light and will eventually stop when the closest light is RED; in all other cases (GREEN, UNKNOWN) the car will continue to move on within given speed limit.

For Details on Traffic light detection see the section bellow "2.1.1. Traffic Light detection"

#### 2.2 Waypoint updater Node

The purpose of this node is to update the target velocity property of each waypoint based on traffic light and obstacle detection data. This node will subscribe to the /base_waypoints, /current_pose, /obstacle_waypoint, and /traffic_waypoint topics, and publish a list of waypoints ahead of the car with target velocities to the /final_waypoints topic. Here we limite the lookahead waypoint into 50 points and was working with no issues.

![alt text][image11]

* waypoint updater node takes into consideration of a closest traffic light published into  /traffic_waypoint node,
  for yellow/red light it will slow down the vehicle and move to a full stop for a Red light while trying not to exceed the allowed jerk limit, it is defined as MAX_DECEL = 0.5 m/s^3  

* if the near by traffic light state is green or there is no traffic light the car will continue with regular speed while
  not exceeding the maximum allowed velocity limit in Kmph. Maximum allowed velocity is configured in ros/src/waypoint_loader.launch and in ros/src/waypoint_loader_site.launch and is retrieved as ros param from /waypoint_loader/velocity  by this node and update the lookahead waypoint's velocity accordingly  

#### 2.3  Autoware Node

A package containing code from Autoware which subscribes to /final_waypoints and publishes target vehicle linear and angular velocities in the form of twist commands to the /twist_cmd topic.
![alt text][image13]


#### 2.4 Drive by Wire(DBW) Node

Carla is equipped with a drive-by-wire (dbw) system, meaning the throttle, brake, and steering have electronic control. This package contains the files that are responsible for control of the vehicle: the node dbw_node.py and the file twist_controller.py, along with a pid and lowpass filter is used to finally drive the car. The dbw_node subscribes to the /current_velocity topic along with the /twist_cmd topic to receive target linear and angular velocities. 

![alt text][image12]

Additionally, this node will subscribe to /vehicle/dbw_enabled, which indicates if the car is under dbw or driver control. This node will publish throttle, brake, and steering commands to the /vehicle/throttle_cmd, /vehicle/brake_cmd, and /vehicle/steering_cmd topics.

For details on DBW and the PID controller see sectin "2.4.1. Drive by Wire(DBW) and PID Controller"

### 2.1.1 Traffic Light detection
This involves  
- Training  
  - Data collection  
    - Given rosbag data was used to capture images for training 
    - Label the data
    - select a proper Model (Meta-architecture and Feature Extractor) to train with the Label images
    - export the inference graph
- Classification
    - Get the realtime image from Carla's Camera
    - Use the frozen inference graph above to classify if there is a traffic light in the image and what is the color of this light.

#### 2.1.1.a Training

##### 2.1.1.a.1 Data Preparation

We got the images of traffic light captured by simulator's camera and that by Carla's(Udacity's self driving car) rosbag [here](https://drive.google.com/file/d/0B2_h37bMVw3iYkdJTlRSUlJIamM/view). We placed simulator and real car's images under `sim_data` and `real_data` folder respectively. Further divided each of them into `train` and `test` folders with 30% images in `test` folder. 

![alt text][image1]

This dataset had a lot of redundant images, meaning, the images were very similar to each other. A series of similar images is shown below:

![alt text][image6]

These images wouldnt add any "more" information and a model wouldnt learn anything new from these similar images. So we decided to remove them from the database.

We used [labelImg](https://github.com/tzutalin/labelImg) to draw bounding box around the traffic light object(s) in the images and label them as either `red` or `yellow` or `green` or `unknown`

Labelling `real` images:

![alt text][image2]

Labelling `sim` images:

![alt text][image3]

labelImg creates `.xml` for each image with information of the image itself and details of the bounding box and its label:

![alt text][image4]

Using a helper function [`xml_to_csv.py`](https://github.com/datitran/raccoon_dataset/blob/master/xml_to_csv.py), all the `.xml` are merged into a single `.csv` file. Another helper function [`generate_tfrecord.py`](https://github.com/datitran/raccoon_dataset/blob/master/generate_tfrecord.py) is used to generate the TFRecord `.record` file which is the file format needed by TensorFlow. Slight modifications were made to the above two helper functions and are placed in this repository.

##### 2.1.1.a.2 Data Exploration

The dataset has 367 1368x1096 `real` images and 280 800x600 `sim` images out of which 30% of each is used as `test` images. The `real` images maynot have any traffic light object or have only one traffic light object. The `sim` images too maynot have any traffic light object or have one or more traffic light object. So the number of objects for classification is 347 `real` objects and 867 `sim` objects. Below is the distribution of objects based on label:

![alt text][image5]

As seen from the above image, the distribution is not same. The `real` data has more of `green` label and the `sim` data has more of `red` label. This imbalance in the dataset will lead the model to be biased towards the label which is more in number. This issue can be fixed by adding more of those images with labels which are less in number. While adding the images, care should be taken not to add similar images but to add images with translation and/or images with different brightness/saturation/hue etc.

##### 2.1.1.a.3 Model Selection

We tried 3 models from the [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). These models are pre-trained on the [COCO dataset](http://mscoco.org/), the [Kitti dataset](http://www.cvlibs.net/datasets/kitti/), and the [Open Images dataset](https://github.com/openimages/dataset). The 3 models we choose are ssd_mobilenet_v1_coco, ssd_inception_v2_coco and faster_rcnn_resnet101_coco. 

First we tarined the `sim` data on the ssd_mobilenet_v1_coco model as it is the lightest and fastest model among the other models in the Tensorflow detection model zoo. The loss graph is as shown below:

![alt text][image7]

The model was trained for 5000 steps which took around 7hrs and the TotalLoss was ~0.35. Some results are shown below:

![alt text][image8]

Next we trained the `real` data on same model for 5000 steps and the loss graph is as below:

![alt text][image9]

The TotalLoss was really bad and was around ~7.5 and this model couldnt classify any objects.  

We tested the SSD_INCEPTION, SSD_MOBILE_V2 and FASTER_RCNN frozen models supplied by Tensorflow, and eventually settled for the F-RCNN model due to its remarkable speed and accuracy

For more detail, see [F-RCNN model for Intelli-Car](https://github.com/Intelli-Car/CarND-Capstone/tree/Hanbyul/ros/src/tl_detector/README.md)

##### 2.1.1.a.4 Training and exporting for inference

All the required steps for training and exporting the inference graph are describe details  [here](https://github.com/Intelli-Car/CarND-Capstone/tree/Hanbyul/ros/src/tl_detector/light_classification/README.md)

#### 2.1.1.b. Classification

tl_detector.py is subscribed for topics /image_color, which provides an image stream from the car's camera. These images are used to determine the color of upcoming traffic lights. Based on Camera encoding this images is converted into RGB i.e. (cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)) before feed into for classification.  

The classification module, tl_classifier.py is based on [CarND Object Detection Lab](https://github.com/udacity/CarND-Object-Detection-Lab)  A new ros parameter named 'model_path' has introduced in tl_detector/launch/tl_detector.launch and in tl_detector/launch/tl_detector_site.launch to locate the frozen inference graph at runtime. By default it is set to FRCNN graph for both Simulator and Real track i.e. tl_detector/light_classification/frozen_models/sim/faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb and tl_detector/light_classification/frozen_models/real/faster_rcnn_resnet101_coco_2018_01_28/frozen_inference_graph.pb in corresponding launch file above. Inference graph for ssd_moobile_v2 or ssd_inception_v2 can be found [here](https://drive.google.com/drive/folders/1hhoOHrnMFiirXvyDQtzZxWZxZHVLD3lr)

### 2.4.a. Drive by Wire(DBW) and PID Controller

Once messages are being published to /final_waypoints, the vehicle's waypoint follower publishes twist commands to the /twist_cmd topic. The drive-by-wire node (dbw_node.py) subscribe to /twist_cmd and use various controllers to provide appropriate throttle, brake, and steering commands. These commands can then be published to the following topics:  

/vehicle/throttle_cmd  
/vehicle/brake_cmd  
/vehicle/steering_cmd  

#### 2.4.1.a PID 

We use the given simple pid controller(pid.py) was use to smoothly drive the car in both track, autonomously.

Since a safety driver may take control of the car during testing, we can not assume that the car is always following the PID commands. If a safety driver does take over, the PID controller will mistakenly accumulate error. For this reason dbw_node.py is subscribed to ros topic /vehicle/dbw_enabled and when this is off, it resets the PID controller's error function to zero.

P Component  

P controller steers the car proportion(inversely, Tau) to Cross Track Error. In this project where the car is driven by the waypoints, CTE was simulated as  

error = Desired velocity - current velocity of the car at a time  

If this number is negative means we want to reduce this error.  

In other word if the above is positive, car drives faster and if negative, car slows down. With a higher Tau the car will oscillate faster.

We used Kp as 0.4, other variation we tried is 0.8 with no significant impact 

D Component  

In PD-controller, for this prooject, when the car is reducing the speed to reduce the error, it won’t just go shooting for the reference velocity but it will notice that it is already reducing the error and as the error is becoming smaller overtime, it counter steers i.e it steers up again, this will allow it to gracefully approach the reference velocity

We do not use the D controller  here so Kd was set to zero.

I Component  

A car might have a form of wheels not aligned appropriately, in robotics it is called Systematic bias. Systematic bias will significantly increase the CTE in PD controller, so the differential term will not be able to compensate for this. This is where the I components come to play which is measured by the integral or the sum of the CTEs over time. Integral coefficient (Ki) should be carefully optimized in small steps as it has a large impact on the overall performance. We use a small value for ki = 0.1 based on trail and error  

Low pass filter  
An average of current velocities are used to derive the current velocity in order to avoid any sporadic spike.  


### 3. References

[Speed/accuracy trade-offs for modern convolutional object detectors](https://arxiv.org/pdf/1611.10012.pdf)  
[SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf)  
[MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf)  
[TensorFlow Models](https://github.com/tensorflow/models)  
[The PASCAL Visual Object Classes Homepage](http://host.robots.ox.ac.uk/pascal/VOC/)  
[udacity/CarND-Object-Detection-Lab](https://github.com/udacity/CarND-Object-Detection-Lab)   

## Installation

Please use **one** of the two installation options, either native **or** docker installation.

### Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

### Docker Installation
[Install Docker](https://docs.docker.com/engine/installation/)

Build the docker container
```bash
docker build . -t capstone
```

Run the docker file
```bash
docker run -p 4567:4567 -v $PWD:/capstone -v /tmp/log:/root/.ros/ --rm -it capstone
```

### Port Forwarding
To set up port forwarding, please refer to the [instructions from term 2](https://classroom.udacity.com/nanodegrees/nd013/parts/40f38239-66b6-46ec-ae68-03afd8a601c8/modules/0949fca6-b379-42af-a919-ee50aa304e6a/lessons/f758c44c-5e40-4e01-93b5-1a82aa4e044f/concepts/16cf4a78-4fc7-49e1-8621-3450ca938b77)

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator

### Real world testing
1. Download [training bag](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/traffic_light_bag_file.zip) that was recorded on the Udacity self-driving car.
2. Unzip the file
```bash
unzip traffic_light_bag_file.zip
```
3. Play the bag file
```bash
rosbag play -l traffic_light_bag_file/traffic_light_training.bag
```
4. Launch your project in site mode
```bash
cd CarND-Capstone/ros
roslaunch launch/site.launch
```
5. Confirm that traffic light detection works on real life images
