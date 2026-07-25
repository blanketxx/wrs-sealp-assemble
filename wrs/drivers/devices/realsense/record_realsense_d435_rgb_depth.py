"""
Record Intel RealSense D435 RGB + Depth video.

Outputs:
    *_rgb.mp4          RGB color video
    *_depth.mp4        Colorized depth visualization video
    *_rgb_depth.mp4    Side-by-side RGB + depth visualization video

Requirements:
    pip install pyrealsense2 opencv-python numpy

Usage:
    python record_realsense_d435_rgb_depth.py

Controls:
    Q / ESC : stop recording and exit

Notes:
    - The D435 depth stream is captured as Z16 internally.
    - *_depth.mp4 is a colorized visualization of the depth stream,
      suitable for viewing/sharing. It is NOT the original 16-bit depth data.
"""

from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs


WIDTH = 640
HEIGHT = 480
FPS = 30

OUTPUT_DIR = Path("realsense_recordings")


def create_writer(path: Path, width: int, height: int, fps: int) -> cv2.VideoWriter:
    """Create an MP4 VideoWriter and verify that it opened successfully."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(
            f"Cannot create video file: {path}\n"
            "Please check the output path or OpenCV video codec support."
        )

    return writer


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    rgb_path = OUTPUT_DIR / f"{timestamp}_rgb.mp4"
    depth_path = OUTPUT_DIR / f"{timestamp}_depth.mp4"
    combined_path = OUTPUT_DIR / f"{timestamp}_rgb_depth.mp4"

    pipeline = rs.pipeline()
    config = rs.config()

    # RGB stream
    config.enable_stream(
        rs.stream.color,
        WIDTH,
        HEIGHT,
        rs.format.bgr8,
        FPS,
    )

    # Depth stream.
    # Z16 stores one unsigned 16-bit depth value per pixel.
    config.enable_stream(
        rs.stream.depth,
        WIDTH,
        HEIGHT,
        rs.format.z16,
        FPS,
    )

    # RealSense colorizer converts the Z16 depth image into a visible color map.
    colorizer = rs.colorizer()

    rgb_writer = None
    depth_writer = None
    combined_writer = None

    try:
        print("Starting Intel RealSense D435...")
        profile = pipeline.start(config)

        # Read depth scale for reference.
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        print(f"Depth scale: {depth_scale} meter/unit")

        rgb_writer = create_writer(
            rgb_path,
            WIDTH,
            HEIGHT,
            FPS,
        )

        depth_writer = create_writer(
            depth_path,
            WIDTH,
            HEIGHT,
            FPS,
        )

        combined_writer = create_writer(
            combined_path,
            WIDTH * 2,
            HEIGHT,
            FPS,
        )

        print("\nRecording started.")
        print("Press Q or ESC to stop.\n")
        print(f"RGB video:       {rgb_path.resolve()}")
        print(f"Depth video:     {depth_path.resolve()}")
        print(f"RGB + Depth:     {combined_path.resolve()}\n")

        while True:
            frames = pipeline.wait_for_frames()

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # RGB image: H x W x 3, uint8 BGR.
            color_image = np.asanyarray(color_frame.get_data())

            # Original depth data: H x W, uint16.
            # Kept here so it can be used later if raw depth processing is added.
            depth_raw = np.asanyarray(depth_frame.get_data())

            # Convert depth to a viewable pseudo-color image.
            depth_color_frame = colorizer.colorize(depth_frame)
            depth_color_image = np.asanyarray(depth_color_frame.get_data())

            # Ensure the displayed depth image matches RGB size.
            if (
                depth_color_image.shape[1] != WIDTH
                or depth_color_image.shape[0] != HEIGHT
            ):
                depth_color_image = cv2.resize(
                    depth_color_image,
                    (WIDTH, HEIGHT),
                    interpolation=cv2.INTER_NEAREST,
                )

            combined = np.hstack(
                (
                    color_image,
                    depth_color_image,
                )
            )

            # Save videos.
            rgb_writer.write(color_image)
            depth_writer.write(depth_color_image)
            combined_writer.write(combined)

            # Preview labels are not written into the original RGB/depth videos.
            preview = combined.copy()

            cv2.putText(
                preview,
                "RGB",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                preview,
                "DEPTH",
                (WIDTH + 15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Show the center-pixel depth value for quick verification.
            center_depth_m = depth_raw[HEIGHT // 2, WIDTH // 2] * depth_scale

            cv2.putText(
                preview,
                f"Center: {center_depth_m:.3f} m",
                (WIDTH + 15, HEIGHT - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "RealSense D435 - RGB | Depth   (Q/ESC to stop)",
                preview,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

    except RuntimeError as exc:
        print(f"\nRealSense error: {exc}")

    finally:
        if rgb_writer is not None:
            rgb_writer.release()

        if depth_writer is not None:
            depth_writer.release()

        if combined_writer is not None:
            combined_writer.release()

        try:
            pipeline.stop()
        except RuntimeError:
            pass

        cv2.destroyAllWindows()

        print("\nRecording stopped.")
        print("Generated files:")
        print(f"  {rgb_path.resolve()}")
        print(f"  {depth_path.resolve()}")
        print(f"  {combined_path.resolve()}")


if __name__ == "__main__":
    main()
