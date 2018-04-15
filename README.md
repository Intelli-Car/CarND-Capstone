This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).


[//]: # (Image References)

[image1]: ./readme_images/dataset_folder_structure.jpg "Dataset Folder Structure"
[image2]: ./readme_images/labelImg_real_image.jpg "Label Real Image"
[image3]: ./readme_images/labelImg_sim_image.jpg "Label Sim Image"
[image4]: ./readme_images/labelImg_xml_real_image.jpg "XML Real Image"
[image5]: ./readme_images/object_distribution.jpg "Object Distribution"
[image6]: ./readme_images/similar_image_series.jpg "Similar Image Series"
[image7]: ./readme_images/loss_graph_ssd_mobilenet_v1_coco_on_sim_data.jpg "Loss Graph SSD MobilenetV1 On Sim Data"
[image8]: ./readme_images/result_ssd_mobilenet_v1_coco_on_sim_data.jpg "Result SSD MobilenetV1 on Sim Data"

### Data Preparation

We got the images of traffic light captured by simulator's camera and that by Carla's(Udacity's self driving car) camera from <link to the dataset>. We placed simulator and real car's images under `sim_data` and `real_data` folder respectively. Further divided each of them into `train` and `test` folders with 30% images in `test` folder. 

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

### Data Exploration

The dataset has 367 1368x1096 `real` images and 280 800x600 `sim` images out of which 30% of each is used as `test` images. The `real` images maynot have any traffic light object or have only one traffic light object. The `sim` images too maynot have any traffic light object or have one or more traffic light object. So the number of objects for classification is 347 `real` objects and 867 `sim` objects. Below is the distribution of objects based on label:

![alt text][image5]

As seen from the above image, the distribution is not same. The `real` data has more of `green` label and the `sim` data has more of `red` label. This imbalance in the dataset will lead the model to be biased towards the label which is more in number. This issue can be fixed by adding more of those images with labels which are less in number. While adding the images, care should be taken not to add similar images but to add images with translation and/or images with different brightness/saturation/hue etc.

### Model

We tried 3 models from the [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). These models are pre-trained on the [COCO dataset](http://mscoco.org/), the [Kitti dataset](http://www.cvlibs.net/datasets/kitti/), and the [Open Images dataset](https://github.com/openimages/dataset). The 3 models we choose are ssd_mobilenet_v1_coco, ssd_inception_v2_coco and faster_rcnn_resnet101_coco. 

First we tarined on the ssd_mobilenet_v1_coco model as it is the lightest and fastest model among the other models in the Tensorflow detection model zoo. The loss graph is as shown below:

![alt text][image7]

The model was trained for 5000 steps which took around 7hrs and the TotalLoss was ~0.35. Some results are shown below:

![alt text][image8]

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
