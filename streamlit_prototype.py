from contextlib import closing
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

import imageio_ffmpeg
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO


WEIGHTS = Path(__file__).resolve().parent / "weights" / "road_damage_yolo26s.pt"
PREDICT = {"imgsz": 640, "conf": 0.25, "device": "cpu", "verbose": False}


def annotate_image(model, upload):
    upload.seek(0)
    with Image.open(upload) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
    result = model.predict(image, **PREDICT)[0]
    return image, result.plot()[..., ::-1].copy(), len(result.boxes)


def annotate_video(model, upload, status):
    with TemporaryDirectory() as folder:
        source = Path(folder) / "input.mp4"
        destination = Path(folder) / "annotated.mp4"
        source.write_bytes(upload.getbuffer())

        frames_processed = frames_with_boxes = 0
        with closing(imageio_ffmpeg.read_frames(str(source), pix_fmt="bgr24")) as reader:
            metadata = next(reader, None)
            if metadata is None:
                raise ValueError("Could not read the uploaded video.")
            width, height = metadata["size"]
            fps = metadata["fps"]
            if not isfinite(fps) or fps <= 0:
                raise ValueError("Could not read the video's frame rate.")

            with closing(imageio_ffmpeg.write_frames(
                str(destination), (width, height), fps=fps, codec="libx264",
                pix_fmt_in="bgr24", macro_block_size=2, output_params=["-movflags", "+faststart"],
            )) as writer:
                writer.send(None)
                for frame in reader:
                    image = np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 3).copy()
                    result = model.predict(image, **PREDICT)[0]
                    writer.send(np.ascontiguousarray(result.plot()).tobytes())
                    frames_processed += 1
                    frames_with_boxes += len(result.boxes) > 0
                    if frames_processed % 25 == 0:
                        status.write(f"Processing video… {frames_processed} frames checked.")

        if not frames_processed or not destination.is_file() or not destination.stat().st_size:
            raise ValueError("Could not produce an annotated video.")
        return destination.read_bytes(), frames_processed, frames_with_boxes


def clear_result():
    st.session_state.pop("result", None)


def main():
    st.set_page_config(page_title="Road Damage Detection", layout="wide")
    st.title("Road Damage Detection")
    st.write("Upload a road photo or a short dashcam clip to see the damage YOLO26s detects.")
    st.caption("YOLO26s · CPU · image size 640 · confidence threshold 0.25")

    if not WEIGHTS.is_file():
        st.error("Place your trained checkpoint at weights/road_damage_yolo26s.pt.")
        st.stop()

    upload = st.file_uploader(
        "Choose a JPG, PNG, or MP4", type=["jpg", "jpeg", "png", "mp4"],
        max_upload_size=50, key="road_media", on_change=clear_result,
    )
    if upload is None:
        return
    is_video = Path(upload.name).suffix.lower() == ".mp4"

    if st.button("Detect road damage", type="primary"):
        clear_result()
        try:
            with st.spinner("Running YOLO26s…"):
                if "model" not in st.session_state:
                    st.session_state.model = YOLO(str(WEIGHTS))
                model = st.session_state.model
                status = st.empty()
                try:
                    st.session_state.result = (
                        annotate_video(model, upload, status) if is_video
                        else annotate_image(model, upload)
                    )
                finally:
                    status.empty()
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"Could not process this file: {error}")

    result = st.session_state.get("result")
    if result is None:
        return
    if is_video:
        video, frames, with_boxes = result
        st.write(f"Detections in {with_boxes:,} of {frames:,} frames.")
        st.video(video, format="video/mp4")
        st.caption("The annotated video is silent.")
    else:
        original, annotated, count = result
        st.write(f"Detections: {count}")
        left, right = st.columns(2)
        left.image(original, caption="Original", width="stretch")
        right.image(annotated, caption="YOLO26s predictions", width="stretch")


if __name__ == "__main__":
    main()
