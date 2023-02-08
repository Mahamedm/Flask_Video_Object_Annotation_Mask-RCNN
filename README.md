# Flask Video Object Annotation - Mask RCNN
![Screenshot_20230208_020940](https://user-images.githubusercontent.com/52294485/217539563-df8e37f3-f8cd-4fa2-8952-8df2f1bc8dd1.png)
## Introduction
This project is a web-based platform that allows users to annotate objects in videos using Mask RCNN. The platform is built using Flask, a Python-based web framework, and the Mask RCNN deep learning model, developed for object detection and instance segmentation.

## Model
The project uses [Mask R-CNN](https://arxiv.org/abs/1703.06870), a state-of-the-art object instance segmentation and classification model. The model was trained on the [MS-COCO dataset](https://cocodataset.org/#home), which contains 80 different object categories and 330k images.

The MS-COCO dataset provides a diverse set of images for training and validation, making it suitable for various real-world scenarios. The model achieved high accuracy in detecting and segmenting objects from the MS-COCO dataset, with a mAP (mean Average Precision) of 0.544 and an IoU (Intersection over Union) of 0.5 on the validation set.

The training of the model was performed using transfer learning, where a pre-trained model was fine-tuned on the MS-COCO dataset to produce a model optimized for this specific use case. This approach allowed the model to quickly converge to good performance while using a relatively small amount of training data.

## Features
- Video upload and processing
- Object annotation using Mask RCNN

## Requirements
- Python 3.6.12
- Flask
- TensorFlow
- Keras
- Numpy
- OpenCV
## Installation
<!-- Navigate to the project directory:<br>
<code>cd Flask_Video_Object_Annotation_Mask-RCNN</code><br>-->
- Create a virtual environment with conda:
<br><code>conda create -n [environment] python=3.6.12</code><br>
- Activate the virtual environment:
<br><code>conda activate [environment]</code><br>
- Install the required packages:
<br><code>[environment path]/python.exe -m pip install -r requirements.txt</code><br>
## Usage
- Run: <code>app</code> or open <code>flask_app</code> with anaconda | vs-code | google colab.
- Navigate to **[localhost](http://localhost:5000)** in your web browser.
- Upload a video to annotate.
- Wait for the video to process.
- Review the annotation results on download prompt or on root folder.

##### This project is licensed under the MIT License. See the LICENSE file for details.
