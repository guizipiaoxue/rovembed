import argparse
import os
import subprocess
import time

import cv2
import numpy as np
import yaml

try:
    from camera.camera import MultiUsbCamera
except ImportError:
    from camera import MultiUsbCamera


CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "camera", "camera.yaml")
DEFAULT_WIDTH = 320
DEFAULT_HEIGHT = 240
DEFAULT_FPS = 30
KERNEL_CACHE = {}


class FfmpegCamera:
    def __init__(self, device, width, height, fps):
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.frame_size = self.width * self.height * 3
        self.proc = None

    def start(self):
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "v4l2",
            "-input_format",
            "mjpeg",
            "-framerate",
            str(self.fps),
            "-video_size",
            f"{self.width}x{self.height}",
            "-i",
            self.device,
            "-an",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "-",
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self.frame_size * 2,
        )

    def read(self):
        if self.proc is None or self.proc.stdout is None:
            return False, None
        data = self.proc.stdout.read(self.frame_size)
        if len(data) != self.frame_size:
            return False, None
        frame = np.frombuffer(data, dtype=np.uint8).reshape((self.height, self.width, 3))
        return True, frame.copy()

    def stop(self):
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.proc = None


def load_camera_config(path):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    devices = data.get("devices") or ["/dev/video0"]
    return {
        "device": devices[0],
        "width": int(data.get("width", DEFAULT_WIDTH)),
        "height": int(data.get("height", DEFAULT_HEIGHT)),
        "fps": int(data.get("fps", DEFAULT_FPS)),
    }


def min_filter(image, radius):
    kernel_size = radius * 2 + 1
    kernel = KERNEL_CACHE.get(kernel_size)
    if kernel is None:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        KERNEL_CACHE[kernel_size] = kernel
    return cv2.erode(image, kernel)


def underwater_dark_channel(image_float, radius):
    gb_min = cv2.min(image_float[:, :, 0], image_float[:, :, 1])
    return min_filter(gb_min, radius)


def estimate_atmospheric_light(image_float, dark_channel, top_percent=0.0):
    if top_percent <= 0:
        _, _, _, max_loc = cv2.minMaxLoc(dark_channel)
        x, y = max_loc
        return image_float[y, x]

    flat_dark = dark_channel.reshape(-1)
    flat_image = image_float.reshape(-1, 3)
    count = max(1, int(flat_dark.size * top_percent))
    indices = np.argpartition(flat_dark, -count)[-count:]
    brightness = flat_image[indices].sum(axis=1)
    return flat_image[indices[np.argmax(brightness)]]


def estimate_transmission(image_float, atmospheric_light, radius, omega):
    safe_a = np.maximum(atmospheric_light, 1e-6)
    normalized = image_float / safe_a.reshape(1, 1, 3)
    dark_channel = underwater_dark_channel(normalized, radius)
    return 1.0 - omega * dark_channel


def refine_transmission(transmission, blur_size):
    if blur_size <= 1:
        return transmission
    if blur_size % 2 == 0:
        blur_size += 1
    return cv2.GaussianBlur(transmission, (blur_size, blur_size), 0)


class UDCPDehazer:
    def __init__(
        self,
        radius=5,
        omega=0.95,
        min_transmission=0.12,
        blur_size=7,
        top_percent=0.0,
    ):
        self.radius = int(radius)
        self.omega = float(omega)
        self.min_transmission = float(min_transmission)
        self.blur_size = int(blur_size)
        self.top_percent = float(top_percent)

    def dehaze(self, frame):
        image_float = frame.astype(np.float32) / 255.0
        dark_channel = underwater_dark_channel(image_float, self.radius)
        atmospheric_light = estimate_atmospheric_light(
            image_float,
            dark_channel,
            top_percent=self.top_percent,
        )
        transmission = estimate_transmission(
            image_float,
            atmospheric_light,
            self.radius,
            self.omega,
        )
        transmission = refine_transmission(transmission, self.blur_size)
        transmission = np.maximum(transmission, self.min_transmission)

        recovered = (image_float - atmospheric_light.reshape(1, 1, 3)) / transmission[:, :, None]
        recovered += atmospheric_light.reshape(1, 1, 3)
        recovered = np.clip(recovered, 0.0, 1.0)
        return (recovered * 255).astype(np.uint8)


def udcp_dehaze(frame, radius=5, omega=0.95, min_transmission=0.12, blur_size=7, top_percent=0.0):
    dehazer = UDCPDehazer(
        radius=radius,
        omega=omega,
        min_transmission=min_transmission,
        blur_size=blur_size,
        top_percent=top_percent,
    )
    return dehazer.dehaze(frame)


def resize_for_preview(frame, max_width):
    if max_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (max_width, max(1, int(h * scale))))


def parse_args():
    parser = argparse.ArgumentParser(description="Use camera.py to capture video and apply UDCP dehazing.")
    parser.add_argument("--device", default="", help="Camera device path. Defaults to the first device in camera.yaml.")
    parser.add_argument("--width", type=int, default=0, help="Capture width. Defaults to 320 for real-time dehazing.")
    parser.add_argument("--height", type=int, default=0, help="Capture height. Defaults to 240 for real-time dehazing.")
    parser.add_argument("--fps", type=int, default=0, help="Capture FPS. Defaults to camera.yaml.")
    parser.add_argument("--backend", choices=("ffmpeg", "opencv"), default="ffmpeg", help="Video capture backend.")
    parser.add_argument("--radius", type=int, default=5, help="UDCP dark-channel radius.")
    parser.add_argument("--omega", type=float, default=0.95, help="UDCP haze removal strength.")
    parser.add_argument("--min-transmission", type=float, default=0.12, help="Lower bound for transmission.")
    parser.add_argument("--blur-size", type=int, default=7, help="Gaussian blur size for transmission refinement.")
    parser.add_argument("--top-percent", type=float, default=0.0, help="Atmospheric-light top percent; 0 uses fastest max pixel.")
    parser.add_argument("--process-width", type=int, default=320, help="Resize frames before UDCP; use 0 for full size.")
    parser.add_argument("--preview-width", type=int, default=800, help="Preview width per image.")
    parser.add_argument("--headless", action="store_true", help="Do not show preview windows; print processed frame info.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many processed frames; 0 means run until interrupted.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_camera_config(CONFIG_PATH)

    device = args.device or cfg["device"]
    width = args.width or min(cfg["width"], DEFAULT_WIDTH)
    height = args.height or min(cfg["height"], DEFAULT_HEIGHT)
    fps = args.fps or cfg["fps"]

    manager = None
    ffmpeg_camera = None
    if args.backend == "ffmpeg":
        ffmpeg_camera = FfmpegCamera(device, width, height, fps)
        ffmpeg_camera.start()
    else:
        manager = MultiUsbCamera([device], width=width, height=height, fps=fps)
        cameras = manager.start_all()
        if not cameras:
            print("No camera connected.")
            raise SystemExit(1)

    if args.headless:
        print(f"UDCP dehaze started with {args.backend} backend in headless mode.")
    else:
        print(f"UDCP dehaze preview started with {args.backend} backend. Press 'q' to quit.")
    last_time = time.time()
    frame_count = 0
    total_frames = 0
    display_fps = 0.0
    last_capture_time = 0.0
    dehazer = UDCPDehazer(
        radius=args.radius,
        omega=args.omega,
        min_transmission=args.min_transmission,
        blur_size=args.blur_size,
        top_percent=args.top_percent,
    )

    try:
        while True:
            if ffmpeg_camera is not None:
                ok, frame = ffmpeg_camera.read()
                if not ok:
                    print("Failed to read frame from FFmpeg camera.")
                    break
            else:
                latest = manager.get_latest_frames()
                if device not in latest:
                    time.sleep(0.01)
                    continue

                capture_time, frame = latest[device]
                if capture_time == last_capture_time:
                    time.sleep(0.001)
                    continue
                last_capture_time = capture_time

            process_frame = resize_for_preview(frame, args.process_width)
            dehazed = dehazer.dehaze(process_frame)

            frame_count += 1
            total_frames += 1
            now = time.time()
            if now - last_time >= 1.0:
                display_fps = frame_count / (now - last_time)
                frame_count = 0
                last_time = now
                if args.headless:
                    h, w = dehazed.shape[:2]
                    mean_bgr = dehazed.mean(axis=(0, 1))
                    print(
                        f"frame={total_frames} size={w}x{h} "
                        f"fps={display_fps:.1f} "
                        f"mean_bgr=({mean_bgr[0]:.1f},{mean_bgr[1]:.1f},{mean_bgr[2]:.1f})"
                    )

            if not args.headless:
                original_preview = resize_for_preview(process_frame, args.preview_width)
                dehazed_preview = resize_for_preview(dehazed, args.preview_width)
                cv2.putText(
                    dehazed_preview,
                    f"UDCP {display_fps:.1f} fps",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                preview = np.hstack((original_preview, dehazed_preview))
                cv2.imshow("Original | UDCP Dehazed", preview)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames > 0 and total_frames >= args.max_frames:
                break
    finally:
        if manager is not None:
            manager.stop_all()
        if ffmpeg_camera is not None:
            ffmpeg_camera.stop()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
