#!/usr/bin/env python3
"""
Track 04 - MP4 Ingestion and Streaming Decoder

Responsibility:
    MP4 input -> validation -> one-pass frame decoding -> downstream consumer

This module deliberately does NOT:
    - run EPIC SlowFast or any other encoder
    - classify cooking videos
    - detect actions
    - retain the entire video in RAM

Install:
    pip install opencv-python

CLI:
    python video_ingest.py input.mp4
    python video_ingest.py input.mp4 --output-json metadata.json

Python integration:
    from video_ingest import VideoIngestor

    ingestor = VideoIngestor("input.mp4")
    for frame in ingestor.frames():
        # frame.rgb is a uint8 RGB image (H, W, 3)
        # Pass it to the next stage here.
        process_frame(frame.rgb)

    print(ingestor.metadata)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


LOG = logging.getLogger("video_ingest")


@dataclass
class VideoMetadata:
    file_path: str
    file_name: str
    file_size_bytes: int
    fps: float
    width: int
    height: int
    reported_frame_count: int
    duration_seconds: Optional[float]
    decoded_frame_count: int = 0
    decoding_time_seconds: float = 0.0
    status: str = "not_started"


@dataclass
class DecodedFrame:
    index: int
    timestamp_seconds: float
    rgb: np.ndarray


class VideoIngestor:
    """
    Opens and decodes an MP4 sequentially.

    Frames are yielded one at a time, so memory usage does not grow with
    video duration. Each decoded frame is converted from OpenCV's BGR
    format to RGB before being handed to the downstream pipeline.

    Use frames() only once per instance: it represents one streaming pass.
    """

    def __init__(self, video_path: str | Path):
        self.path = Path(video_path).expanduser().resolve()

        if not self.path.exists():
            raise FileNotFoundError(f"Video file does not exist: {self.path}")

        if not self.path.is_file():
            raise ValueError(f"Input path is not a file: {self.path}")

        if self.path.suffix.lower() != ".mp4":
            raise ValueError(
                f"Expected an MP4 file, received: {self.path.suffix or 'no extension'}"
            )

        if self.path.stat().st_size == 0:
            raise ValueError("Input MP4 is empty.")

        self.metadata = self._read_metadata()
        self._consumed = False

    def _read_metadata(self) -> VideoMetadata:
        """Read container/stream properties without decoding the full video."""
        capture = cv2.VideoCapture(str(self.path))

        if not capture.isOpened():
            capture.release()
            raise ValueError(
                "OpenCV could not open this MP4. The file may be corrupted "
                "or its video codec may be unsupported."
            )

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

            if width <= 0 or height <= 0:
                raise ValueError("Invalid video dimensions.")

            # Some containers/codecs do not report FPS or frame count reliably.
            if not np.isfinite(fps) or fps < 0:
                fps = 0.0

            duration = (
                frame_count / fps
                if fps > 0 and frame_count > 0
                else None
            )

            return VideoMetadata(
                file_path=str(self.path),
                file_name=self.path.name,
                file_size_bytes=self.path.stat().st_size,
                fps=fps,
                width=width,
                height=height,
                reported_frame_count=frame_count,
                duration_seconds=duration,
            )
        finally:
            capture.release()

    def frames(self) -> Iterator[DecodedFrame]:
        """
        Decode the MP4 in one sequential pass.

        Yields:
            DecodedFrame:
                index: zero-based frame index
                timestamp_seconds: approximate timestamp based on FPS
                rgb: decoded uint8 RGB frame with shape (height, width, 3)

        The caller should consume this generator directly instead of
        converting it to list(), which would load the entire video into RAM.
        """
        if self._consumed:
            raise RuntimeError(
                "This VideoIngestor has already been consumed. "
                "Create a new VideoIngestor instance for another pass."
            )

        self._consumed = True
        capture = cv2.VideoCapture(str(self.path))

        if not capture.isOpened():
            self.metadata.status = "error"
            raise RuntimeError("Failed to reopen video for decoding.")

        frame_index = 0
        start_time = time.perf_counter()
        self.metadata.status = "decoding"

        try:
            while True:
                success, frame_bgr = capture.read()

                if not success:
                    break

                if frame_bgr is None or frame_bgr.size == 0:
                    LOG.warning("Skipping invalid frame at index %d", frame_index)
                    frame_index += 1
                    continue

                # OpenCV decodes in BGR; downstream models commonly expect RGB.
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                if self.metadata.fps > 0:
                    timestamp = frame_index / self.metadata.fps
                else:
                    timestamp = 0.0

                packet = DecodedFrame(
                    index=frame_index,
                    timestamp_seconds=timestamp,
                    rgb=frame_rgb,
                )

                frame_index += 1
                self.metadata.decoded_frame_count = frame_index

                # Yield immediately: do not accumulate frames in memory.
                yield packet

        except Exception:
            self.metadata.status = "error"
            raise

        finally:
            capture.release()
            self.metadata.decoding_time_seconds = round(
                time.perf_counter() - start_time, 4
            )

        if self.metadata.decoded_frame_count == 0:
            self.metadata.status = "error"
            raise RuntimeError("No frames could be decoded from the MP4.")

        self.metadata.status = "success"

        # Replace unreliable container frame count with actual decoded count
        # only in decoded_frame_count; preserve reported_frame_count separately.

    def get_metadata(self) -> dict:
        """Return current metadata as a JSON-serializable dictionary."""
        return asdict(self.metadata)


def run_cli(video_path: str, output_json: Optional[str] = None) -> int:
    """Run ingestion from the terminal without storing decoded frames."""
    try:
        ingestor = VideoIngestor(video_path)

        print("\nVIDEO INFORMATION")
        print("-----------------")
        print(f"File       : {ingestor.metadata.file_name}")
        print(f"Resolution : {ingestor.metadata.width} x {ingestor.metadata.height}")
        print(f"FPS        : {ingestor.metadata.fps}")
        print(f"Reported frames: {ingestor.metadata.reported_frame_count}")
        print(f"Duration   : {ingestor.metadata.duration_seconds} seconds")
        print("\nStarting one-pass decoding...\n")

        # Consume every frame once. In the integrated pipeline, the next
        # stage's frame-processing function will be called at this point.
        for frame in ingestor.frames():
            if frame.index % 100 == 0:
                print(
                    f"\rDecoded {frame.index + 1} frames "
                    f"(timestamp: {frame.timestamp_seconds:.2f}s)",
                    end="",
                    flush=True,
                )

        print("\n\nINGESTION COMPLETE")
        print("------------------")

        result = ingestor.get_metadata()
        print(json.dumps(result, indent=4))

        if output_json:
            output_path = Path(output_json).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=4), encoding="utf-8")
            print(f"\nMetadata saved to: {output_path}")

        return 0

    except (FileNotFoundError, ValueError, RuntimeError) as error:
        error_result = {
            "status": "error",
            "file_path": str(Path(video_path).expanduser()),
            "error": str(error),
        }
        print(json.dumps(error_result, indent=4), file=sys.stderr)
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Validate and decode an MP4 video in one streaming pass."
    )
    parser.add_argument("input", help="Path to input MP4 file")
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save video metadata as JSON",
    )
    args = parser.parse_args()

    sys.exit(run_cli(args.input, args.output_json))


if __name__ == "__main__":
    main()
