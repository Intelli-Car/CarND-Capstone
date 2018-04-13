# Traffic light detection 

## Result 
Refer [notebook](./tl_detection.ipynb)

## Commands for training and exporting for inference
For copy and paste. :)

Needs dataset of TFRecord format and label map of pbtxt type.


```
# From tensorflow/models/research/
# Add Libraries to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
# Testing the Installation
python object_detection/builders/model_builder_test.py
```

#### faster_rcnn_resnet101_coco_2018_01_28
##### Train
```
# From tensorflow/models/research/
# for real data
python object_detection/train.py --pipeline_config_path config/real/faster_rcnn_resnet101_tl.config --train_dir train_dir/real/faster_rcnn_resnet101_coco_2018_01_28
# for simulator data
python object_detection/train.py --pipeline_config_path config/sim/faster_rcnn_resnet101_tl.config --train_dir train_dir/sim/faster_rcnn_resnet101_coco_2018_01_28
```
##### Export frozen models
```
# From tensorflow/models/research/
# for real data
python object_detection/export_inference_graph.py --pipeline_config_path config/real/faster_rcnn_resnet101_tl.config --trained_checkpoint_prefix train_dir/real/faster_rcnn_resnet101_coco_2018_01_28/model.ckpt-87789 --output_directory frozen_models/real/faster_rcnn_resnet101_coco_2018_01_28
# for simulator data
python object_detection/export_inference_graph.py --pipeline_config_path config/faster_rcnn_resnet101_tl.config --trained_checkpoint_prefix train_dir/sim/faster_rcnn_resnet101_coco_2018_01_28/model.ckpt- --output_directory frozen_models/sim/faster_rcnn_resnet101_coco_2018_01_28
```

#### ssd_inception_v2_coco_2017_11_17
```
# From tensorflow/models/research/
# for real data
python object_detection/train.py --pipeline_config_path config/real/ssd_inception_v2_tl.config --train_dir train_dir/real/ssd_inception_v2_coco_2017_11_17
# for simulator data
python object_detection/train.py --pipeline_config_path config/sim/ssd_inception_v2_tl.config --train_dir train_dir/sim/ssd_inception_v2_coco_2017_11_17
```

```
# From tensorflow/models/research/
# for real data
python object_detection/export_inference_graph.py --pipeline_config_path config/real/ssd_inception_v2_tl.config --trained_checkpoint_prefix train_dir/real/ssd_inception_v2_coco_2017_11_17/model.ckpt-{#####} --output_directory frozen_models/real/ssd_inception_v2_coco_2017_11_17
# for simulator data
python object_detection/export_inference_graph.py --pipeline_config_path config/sim/ssd_inception_v2_tl.config --trained_checkpoint_prefix train_dir/sim/ssd_inception_v2_coco_2017_11_17/model.ckpt-{#####} --output_directory frozen_models/sim/ssd_inception_v2_coco_2017_11_17
```

#### ssd_mobilenet_v2_coco_2018_03_29
```
# From tensorflow/models/research/
# for real data
python object_detection/train.py --pipeline_config_path config/real/ssd_mobilenet_v2_tl.config --train_dir train_dir/real/ssd_mobilenet_v2_coco_2018_03_29
# for simulator data
python object_detection/train.py --pipeline_config_path config/sim/ssd_mobilenet_v2_tl.config --train_dir train_dir/sim/ssd_mobilenet_v2_coco_2018_03_29
```

```
# From tensorflow/models/research/
# for real data
python object_detection/export_inference_graph.py --pipeline_config_path config/real/ssd_mobilenet_v2_tl.config.config --trained_checkpoint_prefix train_dir/real/ssd_mobilenet_v2_coco_2018_03_29/model.ckpt-{#####} --output_directory frozen_models/real/ssd_mobilenet_v2_coco_2018_03_29
# for simulator data
python object_detection/export_inference_graph.py --pipeline_config_path config/sim/ssd_mobilenet_v2_tl.config.config --trained_checkpoint_prefix train_dir/sim/ssd_mobilenet_v2_coco_2018_03_29/model.ckpt-{#####} --output_directory frozen_models/sim/ssd_mobilenet_v2_coco_2018_03_29
```

### tensorboard
```
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
tensorboard --logdir=train_dir/real/faster_rcnn_resnet101_coco_2018_01_28
```