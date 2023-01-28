import os
import cv2


def smooth_video(source_path: str, output_path: str):
    cap = cv2.VideoCapture(source_path)

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Get the codec information for the video
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    # Get the frame size of the video
    frame_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    # Create a VideoWriter object to save the output video
    output_path = os.path.join(output_path, "output_interpolated.mp4")
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size, isColor=True)

    # Create a background subtractor object
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    #####################
    ret, prev_frame = cap.read()

    # Read the frames of the input video and write the interpolated frames to the output video
    while True:
        # Read the next frame
        ret, frame = cap.read()
        prev_frame = cv2.resize(prev_frame, (frame_size[0], frame_size[1]))
        frame = cv2.resize(frame, (frame_size[0], frame_size[1]))
        # If there are no more frames, break out of the loop
        if not ret:
            break

        # Perform motion interpolation on the current frame
        fg_mask = bg_subtractor.apply(frame)
        flow = cv2.calcOpticalFlowFarneback(
            prev_frame, frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        interpolated_frame = cv2.remap(frame, flow, None, cv2.INTER_LINEAR)

        # Write the current frame to the output video
        out.write(interpolated_frame)

        prev_frame = frame

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()


smooth_video("dataset/result/eval_video_1.mp4", "dataset/result/")
