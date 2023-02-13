from __future__ import annotations
from dataclasses import dataclass, field

from PaddleDetection.deploy.python.infer import Detector

from video_processor import VideoProcessor


@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.2  # 0.4
    track_buffer: int = 30
    match_thresh: float = 0.7  # 0.7
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False


@dataclass(frozen=True)
class Args:
    BYTE_TRACKER_ARGS = BYTETrackerArgs
    STRIDE: int = 8
    REDUCTION_FACTOR: int = 3
    SOURCE_VIDEO_PATH: str = (
        "data/Hackathon_Stage2/Evaluation_set/video/eval_video_1.mp4"
    )
    TARGET_VIDEO_PATH: str = "/temp/eval_video_1.mp4"
    MODEL_DIR: str = "/home/5qx9nf8a/team_workspace/PaddleDetection/output_inference/ppyoloe_plus_crn_m_80e_coco"
    BATCH_SIZE: int = 32
    RUN_MODE: str = "paddle"
    CPU_THREADS: int = 1
    MAX_DETECTIONS: int = 300
    CLASS_NAMES_DICT: dict = field(
        default_factory=lambda: {
            0: "klt_box_empty",
            1: "klt_box_full",
            2: "rack_1",
            3: "rack_2",
            4: "rack_3",
            5: "rack_4",
            6: "placeholder",
        }
    )

    SCANNER_X: int = 300
    SCANNER_Y: int = 50
    BOX_THRESHOLD: float = 0.35
    RACK_THRESHOLD: float = 0.79


if __name__ == "__main__":
    SOURCE_VIDEO_PATH = "data/live_demo/Evaluation_set/demo_eval_video/full_eval_demo_video.mp4"  # "data/Hackathon_Stage2/Evaluation_set/video/eval_video_1.mp4"
    TARGET_VIDEO_PATH = "/home/5qx9nf8a/team_workspace/temp/"
    args = Args()

    detector = Detector(
        model_dir=args.MODEL_DIR,
        device="GPU",
        run_mode=args.RUN_MODE,
        batch_size=args.BATCH_SIZE,
        trt_min_shape=1,
        trt_max_shape=1280,
        trt_opt_shape=640,
        trt_calib_mode=False,
        cpu_threads=args.CPU_THREADS,
        enable_mkldnn=False,
        enable_mkldnn_bfloat16=False,
        output_dir="output_paddle",
        threshold=0.3,
        delete_shuffle_pass=False,
    )

    video_processor = VideoProcessor(
        source_video_path=SOURCE_VIDEO_PATH, target_dir=TARGET_VIDEO_PATH, args=args
    )
    video_processor.process_video(
        detector, with_scanner=True, with_placeholders=True, with_annotate_scanner=True
    )
    # video_processor.create_submission(mAP=93.3, fps=103.68, save=True)
