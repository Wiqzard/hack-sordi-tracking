

# SOLUTION REPORT 



## Final Solution:
### Data
- Add 5000 images from stage 1, specifically from the Regensburg plant as background images. Make sure there are none with klts or racks in them.
- Used stage_1/transform_data.py to convert the data to the yolov5 format. And the fifyone library to convert from yolov5 to COCO format.
### Detection
 - Use PaddleDetection library for state of the art object detection, that runs under Apache 2.0 Licence
 - As detector use PP-YOLOE + architecture. The size m, to for best mAP/FPS tradeoff.
- Source <https://github.com/PaddlePaddle/PaddleDetection>
- The model was trained on the Nvidia Cluster with an A100(80GB) GPU for 150 epochs, approximately 45 hours. For training used the standard data augmentation provided by the PaddleDetection framework (resizing, mosaic etc.). Input training size was 800x800 pixels, with a batch size of 24.
- Exported the model to tensorrt with PaddleLibrary export.py
- The trained exported model is to be found in the model/ folder.

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
    - Instantiate a Detections class containing the information of the detections of the frame(detections/detection_utils.py)
    - Remove all detections with center of boxes above 600 pixels (from top to bottom)
    - Remove all detections with an bounding box area of less than 2500
    - Remove all rack detections with a confidence less than 0.75 
    - Remove all box detecitons with a confidence of less than 0.35
    - Multiprocessing for the batch 
### TRACKER: (get_tracks())
    - feeds the tracker the detections
    - matches the detectoins with the tracks to receive tracker ids for detections
    - adds the tracker ids to the Detections instance
    - filter out all detections that do not have a tracker id
    - it is import that this gets done in the right seqeuential order for the batches

### SCANNER: (update_scanner(), scanner.update(), found in tracking/rack_counter.py)
    - Our approach is founded on the following core principle that informs our methodology. See below for pictures
    - We employ a scanner that is represented by a vertical line and situated on the left-hand portion of the screen. Our objective is to facilitate the tracker's acquisition of more information concerning the klt and rack detections of frames as objects pass from right to left. This approach instills greater confidence in the tracker's detections as it continues to observe them and helps to mitigate issues with occlusion. Future research could examine other tracker architectures that actively learn about objects. See prospects.
    - As soon as a rack detection passes the scanner, based on the confidence, previous racks and other factors, the scanner sets this rack detection as the current rack and creates a new RackTracker instance (tracking/tracking_counter.py) that stores the information about the rack and its contents for the submission. 
    - After setting the current rack, the scanner awaits new klts. When a klt detection passes through the scanner, we use the find_shelf function that relies on the center of the klt detection bounding box and the handpicked shelf ranges of the current rack's previous frame to locate the shelf that contains the box. Notably, our scanner is stationary, and the shelf ranges remain constant, which simplifies implementation and enhances ccuracy.
<p align="center">
    <img src="utils/imgs/clip.gif " data-canonical-src="utils/imgs/clip.gif" width="600" height="350" />
</p>

### Add placeholders:(core to be found in detections/detection_tools.py get_placeholders_for_racks())
    - Takes all the rack detections that are left to the scanner. Looks up the relative coordinates of all possible klts relative to the rack detection bounding box.
    - Generates all possible klts boxes of that specific rack. 
    - Removes all new klt boxes that have an IoU with the actual detected boxes of greater than 0.25. 
    - Returns a seperable Detections instance with all the possible placeholders.
    - Uses box annotator to annotate the placeholders in the frames

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


