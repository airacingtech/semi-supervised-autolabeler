# Startup
python roar_server.py

# Installation
1. Install Docker Engine (https://docs.docker.com/desktop/install/ubuntu/) NOT Docker Desktop.

# Notes
1. Always save the annotations in the CVAT server before exporting them.
2. Make sure to to use CVAT for video 1.1 and NOT CVAT for images 1.1 when exporting the annotations from the CVAT server.
3. Always reupload the zip folder to the seg and track server after re-exporting the annotations from the CVAT server as the annotations are not updated automatically even if the zip folder has the same name.
4. When uploading the annotations back onto CVAT uncheck "Convert masks to polygons".
5. When exporting a regsementation you don't have to save the images, just the annotations.
6. When exporting the dataset out of CVAT for fine-tuning, select the "COCO 1.0" format.

# Debugging
When debugging be aware that errors thrown will be shown in the web console of the browser (by pressing F12 and going to the console tab). The errors will be shown in the console tab of the browser and not in the terminal where the server is running. 

# TODO
- [ ] Remove BERT
- [ ] Add error message if user uploads a CVAT Images 1.1 file instead of a CVAT Video 1.1 file


# Transform every 100 Key Frames into Thousands of Labeled Images: Labeled Segmented Image Datasets from Video with ROAR-SAMT and CVAT Integration

Transforming video annotation from tedious manual work to semi-supervised automation, the CVAT Instance Segmentation Tracker leverages cutting-edge technology from OpenCV and SAM-Track to enable seamless tracking of segmented objects across video frames. By integrating with CVAT and utilizing SAM-Track's real-time segmentation capabilities, this tool accelerates the creation of labeled datasets from videos, enhancing efficiency and accuracy in object detection training. Ideal for teams needing to scale their annotation processes, it empowers users to automate the creation of thousands of labeled images from a small set of human labeled key frames.

### CVAT Instance Segmentation Tracker

Jump to [running the server](#running-the-server)

## Overview

   OpenCV’s Computer Vision Annotation Tool (https://github.com/opencv/cvat) is an annotation tool that has recently been updated to use Facebook Research’s Segment-Anything-Model (https://github.com/facebookresearch/segment-anything) allows for high quality segmentations to be produced on any given image uploaded into CVAT, including frames of videos, but only allows for single frame annotations. In order to produce segmentation masks to label video data, **each frame** would need to be **done by hand**. CVAT only supports bounding box trackers out of the box (no pun intended). Recently there has been a paper and repository published called SAM-Track (https://github.com/z-x-yang/Segment-and-Track-Anything) which can track segmentation masks real-time, which is super fast and accurate, to extend Instance Segmentation from images to videos!

### Problem

   CVAT allows for team labelling by organizing team structures and roles, and assigning jobs and tasks. It also supports labeling with segmentation masks (i.e. can assign a certain type of mask to be a car or a road) which can be used to produce labeled training data for object detection with segmentation masks rather than bounding box methods to predict not just the location and label of an object, but its shape as well ([Object Detection vs Object Segmentation](https://www.linkedin.com/pulse/object-segmentation-vs-detection-which-one-should-you-ritesh-kanjee/)). 

   The problem is that CVAT does not currently support Segmentation Tracking. We want to track segmented objects throughout a video based on an initial segmentation on any given frame from CVAT. SAM-Track also does not support annotation file imports or labeled segmentation masks. 

Instance Segmentation Tracker solves this problem.

### Bounding Box Tracking vs Instance Segmentation Tracking

![bounding box tracking](docs/img/example-bboxes.png)
![instance segmentation tracking](docs/img/instance_seg.webp)

<aside>
📧 The Instance Segmentation Tracker (IST), built off of (https://github.com/z-x-yang/Segment-and-Track-Anything), allows for seamless and efficient Labeled Instance Segmentation Tracking.
</aside>

Without a [segmentation tracker](https://github.com/airacingtech/roar-seg-and-track-anything/assets/83838942/7431caa6-2c41-4dc6-b89c-4fd6d793c607)
, labeling video data with segmentation masks in CVAT would be
[manually annotated by hand](https://github.com/airacingtech/roar-seg-and-track-anything/assets/83838942/87ed37ac-b8cf-4baa-8928-e24879cabb88)


## Installation

```bash
git clone git@github.com:airacingtech/semi-supervised-autolabeler
cd semi-supervised-autolabeler
conda env create -f updated_environment.yml
conda activate SAMT
pip install -r requirements.txt
```

## Setting up rabbitmq server

### On Ubuntu
```bash
sudo apt update
sudo apt install -y erlang
sudo apt install rabbitmq-server
```

verify service is running

```bash
sudo systemctl status rabbitmq-server.service
```

or if running docker container:

```yml
docker run -d -p 5672:5672 rabbitmq
```


Follow requirements and model preparation via original SAMT github if you want SAM and/or Grounding Dino checkpoints:

[GitHub - z-x-yang/Segment-and-Track-Anything: An open-source project dedicated to tracking and segmenting any objects in videos, either automatically or interactively. The primary algorithms utilized include the Segment Anything Model (SAM) for key-frame segmentation and Associating Objects with Transformers (AOT) for efficient tracking and propagation purposes.](https://github.com/z-x-yang/Segment-and-Track-Anything#bookmark_tabsrequirements)

only need to run:

```bash
bash script/install.sh
bash script/download_ckpt.sh
```

if ./ckpt/R50_DeAOTL_PRE_YTB_DAV.pth does not exist then download the model weights from [here](https://github.com/yoxu515/aot-benchmark) for DeAOT model and place in the ckpt directory.


## Running the server

Make sure to configure your `.env` (see `example.env`).

### Configuring global variables
In roar_config.py, edit the following:
- **PORT** The specified pot to host the server. Defaults to `5000`.
- **HOST** The IP address of the server. Defaults to `localhost`.
- **DOWNLOADS_PATH** The path where annotation zips that need to be tracked are placed. Should ideally be placed where the CVAT server or user places downloaded zips. The paired CVAT Docker repository that has modified to work with this repository does this automatically.
- **DEBUG** Set to true to see debug logs

### Create an agent key
For security purposes on the Flask server, create an `agent.key` file in the repository directory with a string value that is hashed or random

### CUDA Toolkit 12.1
Make sure when you run
```
nvcc --version
```

it returns that your system is running CUDA Toolkit 11.8

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Feb__7_19:32:13_PST_2023
Cuda compilation tools, release 12.1, V12.1.66
Build cuda_12.1.r12.1/compiler.32415258_0
```

if not follow the instructions here: [https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#post-installation-actions) and MAKE SURE you specifiy `--toolkit` when running the runfile installer or you will install the CUDA toolkit and the lower level graphics drivers which will likely break your system with a black screen on boot. If you do this accidentally, go to the Additional Drivers tab in Software & Updates in Ubuntu and select the NVIDIA driver that says (proprietary, tested) and reboot. Then run `nvidia-smi` and ensure it does not return an error.



Start workers in another terminal(s):
```bash
celery -A roar_server.celery worker --loglevel=info -P eventlet -E -n worker1
celery -A roar_server.celery worker --loglevel=info -P eventlet -E -n worker2
...
```
can run `start_celery_worker.sh` to do s automatically.
```bash
./start_celery_worker.sh "worker number"
```

In the root directory:
```bash
conda activate SAMT
python3 roar_server.py
```

## Using the GUI

![gui](docs/img/gui.png)

- **Job**
  - **Initial Segmentation** will take the initial manually annotated frames (uploaded from cvat) and will track these annotations through any frames that aren't manually annotated.
  - **Re-segmentation** is for annotation fixes; select individually and manually fixed frames by entering comma separated numbers
- **Job ID** should match one of the job ids in the `Ready` box
- **Threads** determines max thread workers; useful if multiple different frames are annotated for either inital or resegmentation
- **Automatic Track** will submit the job
- **Frame-by-Frame Tracking** will open panel for tracking through frames by manual input of user. See more details of usage below.
- **Advanced options**
  - **Reuse Annotation Output**
      - if you want to reuse the annotation output from a previous tracking run for this specific job, click yes
      - useful if you want to re-track with the frame right before tracking diverges and you don’t want to reannotate
  - **Delete zip**: will delete the zip file in `DOWNLOADS_PATH` after a successful job execution
### Status Boxes 
There are 5 different status boxes, `Ready`, `Queued Jobs`, `Jobs in Progress`, `Completed Jobs`, and `Failed Jobs`.
- **Ready** Will display newly added job zips that are ready to be tracked. 
  - To use: In the specified `DOWNLOADS_PATH`, make a updates.txt file which should have line separated test specifying annotation zips that exist for tracking e.g 123.zip. The local CVAT Docker solution that pairs with this repository should automatically handle this.
- **Queued Jobs** Will display jobs that are currently waiting to be handed over to a worker.
- **Jobs in Progress** Will display jobs that are currently being tracked.
- **Completed Jobs** Will display jobs that have completed. Click on the job number to download the respective completed annotation zip
- **Failed Jobs** Will display jobs that have failed along with any error specifying why.

### Manual tracking (frame-by-frame)
A visual version of the tracker, can see what the tracker is tracking at each frame and quickly export to CVAT to re-segment once object tracking diverges.

- PROS:
    - can see each frame the tracker is tracking
    - can quickly export annotations once tracking diverges unlike automatic tracking which will track all frames given even if tracking diverges
        - automatic tracking will make annotations that diverge once tracking diverges, usually past a key frame
- CONS
    - Automatic tracking can take advantage of parallel processing if given multiple key frame annotations, Frame-by-Frame is linear and will generate new frames only if specified
- **HOW TO USE:**
   - ![manual tracking](docs/img/gui-frame-by-frame.png)
    - Type in a given frame in the valid range and press enter
    - can use forward and backward to iterate current frame by +1/-1 respectively
    - Demonstration https://github.com/airacingtech/roar-seg-and-track-anything/assets/83838942/d4078540-2677-41d6-9b75-ef6466563bfc


## Command Line Interface

```python
python roar_main.py
```

 
### Credits
* SAM-Track - [https://github.com/z-x-yang/Segment-and-Track-Anything](https://github.com/z-x-yang/Segment-and-Track-Anything


<aside>
contact: Kevin Chow chowmein113@berkeley.edu
</aside>
