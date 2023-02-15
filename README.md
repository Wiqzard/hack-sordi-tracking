

# SOLUTION REPORT v1:



## TOOLS: 
### Data
- In order to mitigate detection errors, we added 1000 images from stage 1, specifically from the Regensburg plant as background images. It was importatnt to make sure that there are no background images with KLT or rack detections included.
- To bring the dataset into the right format for the detection algorithm which utilizes the COCO fromat, we utilized our stage_1 script "stage_1/transform_data.py" to convert the dataset to YOLOv5 format. And following the fifyone library to convert from YOLOv5 to COCO format.
### Detection

- PP-YOLOE+ is a state-of-the-art object detection algorithm that is used in the project. It is implemented using the PaddleDetection library, which is released under the Apache 2.0 license. The PP-YOLOE+ architecture is used as the detector, and the size "m" is selected to achieve the best tradeoff between mean average precision (mAP) and frames per second (FPS).
- Source <https://github.com/PaddlePaddle/PaddleDetection>
- The PP-YOLOE+ model was trained on an Nvidia Cluster with an A100(80GB) GPU for 150 epochs, which took approximately 45 hours. The standard data augmentation techniques provided by the PaddleDetection framework, such as resizing and mosaic, were used during training. The input training size was set to 800x800 pixels, with a batch size of 24.
- The trained model was exported to TensorRT using PaddleLibrary's export.py script, and the exported model can be found in the model/ folder.

### Tracking
- Use ByteTrack as a strong and fast tracking algorithm. Runs under MIT licence.
- Source <https://github.com/ifzhang/ByteTrack>



# Workflow
The control part of the pipleline can be found in video_processor/video_processor.py
To run the pipeline on a video, instantiate a VideoProcessor instance (see main.py) and call the process_video() method.
##  process_video() flow chart:
<p align="center">
    <img src="utils/imgs/flow_png.png " data-canonical-src="utils/imgs/flow_png.png" width="800" height="500" />
</p>

### POSTPROCESS: (initial_results_to_detections())

- Create an instance of the Detections class, which contains the detection information for the current frame (detections/detection_utils.py)
- It removes all detections that have the center of their bounding boxes above 600 pixels from the top to bottom of the frame, as these are likely to be false positives or irrelevant objects that are not part of the rack system we are trying to track.
- It removes all detections with a bounding box area of less than 2500, as these are likely to be noise or clutter in the image that we do not want to include in our annotations.
- Since our approach relies on accurate rack detections, it removes all rack detections with a confidence score of less than 0.75, as these detections are likely to be unreliable or uncertain and may lead to inaccurate annotations. Similar with KLT box detections with a confidence of less than 0.35, since those suffer the most from occlusion.
- It uses multiprocessing for the batch, which allows the processing of multiple frames simultaneously for a speed up.
### TRACKER: (get_tracks())
- Provides the detections to the tracker.
- It matches the detections with the tracks to receive tracker IDs for the detections. This is done to keep track of the same object across multiple frames and ensure consistency in the annotations.
- Appends the tracker IDs to the Detections instance.
- Removes all detections without a tracker ID.
- It is important that this gets done in the right sequential order for the batches, to ensure that the tracker has accurate information about the detections for each frame.
### SCANNER: (update_scanner(), scanner.update(), found in tracking/rack_counter.py)
- Our approach is founded on the following core principle that informs our methodology. See below for pictures
- We employ a scanner that is represented by a vertical line and situated on the left-hand portion of the screen. Our objective is to facilitate the tracker's acquisition of more information concerning the klt and rack detections of frames as objects pass from right to left. This approach instills greater confidence in the tracker's detections as it continues to observe them and helps to mitigate issues with occlusion. Future research could examine other tracker architectures that actively learn about objects. See prospects.
- As soon as a rack detection passes the scanner, based on the confidence, previous racks and other factors, the scanner sets this rack detection as the current rack and creates a new RackTracker instance (tracking/tracking_counter.py) that stores the information about the rack and its contents for the submission. 
- After setting the current rack, the scanner awaits new klts. When a klt detection passes through the scanner, we use the find_shelf function that relies on the center of the klt detection bounding box and the handpicked shelf ranges of the current rack's previous frame to locate the shelf that contains the box. Notably, our scanner is stationary, and the shelf ranges remain constant, which simplifies implementation and enhances ccuracy.
<p align="center">
    <img src="utils/imgs/clip.gif " data-canonical-src="utils/imgs/clip.gif" width="600" height="350" />
</p>

### ADD PLACEHOLDERS:(core to be found in detections/detection_tools.py get_placeholders_for_racks())
- Retrieves all rack detections to the left of the scanner and obtains the relative coordinates of all possible KLTs relative to the rack detection bounding box. 
- First, all rack detections to the left of the scanner are retrieved, and the relative coordinates of all possible KLT boxes with respect to the rack detection bounding box are obtained. Those were hand picked and are stored in constants/bboxes.py
- Generates all potential KLT boxes for the specific rack.
- To filter out new KLT boxes that are irrelevant, a threshold of Intersection over Union (IoU) greater than 0.25 is set, and any potential KLT boxes that exceed this threshold with the actual detected boxes are removed.
- Returns a separate Detections instance containing all possible placeholders.
- Uses the box annotator to annotate the placeholders in the frames.

# Prospects:
- Although we made efforts to optimize our hyperparameters, there remain numerous parameters that we did not have the opportunity to fine-tune. Despite achieving an acceptable solution, we believe that there are superior configurations that could be explored.
- We recommend testing other detectors to improve detection accuracy. For instance, we previously utilized YOLOv8 from ultralytics and observed markedly superior results in the higher confidence domain.
Our scanner-based approach may be better suited to other trackers that leverage deep learning techniques such as DeepSort, especially when performance is not the primary metric.
- To improve placeholder annotations, we must address the issue of the bounding boxes of racks shrinking as they fade out of the image. Since we rely on the relative coordinates of all possible klts in the rack based on the center of the rack detection, this causes placeholders to appear displaced to the right once the rack crosses the left border. We can remedy this by storing the width of the current rack as soon as its left x coordinate touches the left-hand side of the frame and using that width to obtain the actual center of the rack instead of solely the center of the bounding box.
- Another approach to improving placeholder detections could entail gathering the relative coordinates of all potential boxes from more than one rack position. This would foster smoother transitions between different geometries for varying perspectives and reduce the likelihood of errors.
- Our approach is modular in nature, meaning that altering the scanner setup (e.g., moving the camera further back or adjusting its orientation to face in the moving direction to present objects on the left and/or right) merely requires modifying the scanner line and adding the relative coordinates of possible boxes to the CONSTANTS class in constants/bboxes.py.

# Run the solution:
- We reommend looking into the stage2.ipynb notebook for a detailed installation guide. 
- Please ensure that you have installed the necessary packages. Torch, CUDA, and related dependencies are assumed. Installation instructions for PaddleDetection and ByteTrack can be found at:
  - <https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/docs/tutorials/INSTALL.md>
  - <https://github.com/ifzhang/ByteTrack>
- Convert your dataset to COCO format
- In PaddleDetecion/configs/datasets/ add a yml file for your dataset (typical yml format for COCO datasets, see trasnform_2_coco.ipynb for more information)
- In PaddleDetection/configs/ppyoloe/ppyoloe_plus_crn_m_80e_coco.yml, replace the path "'../datasets/coco.yml'" with the path to the YAML file you added for your dataset.
- In the stage2.ipynb notebook or main.py, you will find the necessary parameters for the VideoProcessor class. Instantiate the class and call the process_video method to run the script.


