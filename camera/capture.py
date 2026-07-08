import os
import time
import cv2
import yaml
import numpy as np

from camera import MultiUsbCamera


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "camera.yaml")
ALIGN_WINDOW_SEC = 0.03
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30.0
DEFAULT_CROP_WIDTH = 840
DEFAULT_CROP_HEIGHT = 640


def load_config(path):
	with open(path, "r", encoding="utf-8") as f:
		data = yaml.safe_load(f) or {}
	devices = data.get("devices") or []
	if not devices:
		raise ValueError("No camera devices found in camera.yaml (key: devices)")
	width = int(data.get("width", DEFAULT_WIDTH))
	height = int(data.get("height", DEFAULT_HEIGHT))
	fps = float(data.get("fps", DEFAULT_FPS))

	alignment_data = data.get("alignment") or {}
	alignment = {
		"enabled": bool(alignment_data.get("enabled", False)),
		"source_device": alignment_data.get("source_device", "/dev/video2"),
		"reference_device": alignment_data.get("reference_device", "/dev/video0"),
		"flip_horizontal": bool(alignment_data.get("flip_horizontal", True)),
		"homography": alignment_data.get("homography"),
	}

	crop_data = data.get("crop") or {}
	crop = {
		"width": int(crop_data.get("width", DEFAULT_CROP_WIDTH)),
		"height": int(crop_data.get("height", DEFAULT_CROP_HEIGHT)),
	}

	return devices, width, height, fps, alignment, crop


def center_crop(frame, crop_width, crop_height):
	h, w = frame.shape[:2]
	target_w = max(1, min(int(crop_width), w))
	target_h = max(1, min(int(crop_height), h))
	x0 = (w - target_w) // 2
	y0 = (h - target_h) // 2
	return frame[y0:y0 + target_h, x0:x0 + target_w]


def preprocess_synced_frames(synced_frames, alignment_cfg, crop_cfg, fallback_size):
	processed = {}
	if not synced_frames:
		return processed

	for device, (timestamp, frame) in synced_frames.items():
		processed[device] = (timestamp, frame.copy())

	if alignment_cfg.get("enabled"):
		source_device = alignment_cfg.get("source_device")
		reference_device = alignment_cfg.get("reference_device")
		h_values = alignment_cfg.get("homography")
		if source_device in processed and h_values:
			source_ts, source_frame = processed[source_device]
			if alignment_cfg.get("flip_horizontal", True):
				source_frame = cv2.flip(source_frame, 1)

			if reference_device in processed:
				reference_frame = processed[reference_device][1]
				ref_h, ref_w = reference_frame.shape[:2]
			else:
				ref_w, ref_h = fallback_size

			h_mat = np.array(h_values, dtype=np.float64)
			aligned = cv2.warpPerspective(source_frame, h_mat, (int(ref_w), int(ref_h)))
			processed[source_device] = (source_ts, aligned)

	crop_w = crop_cfg.get("width", DEFAULT_CROP_WIDTH)
	crop_h = crop_cfg.get("height", DEFAULT_CROP_HEIGHT)
	for device, (timestamp, frame) in list(processed.items()):
		processed[device] = (timestamp, center_crop(frame, crop_w, crop_h))

	return processed


def make_session_dir(base_dir):
	ts = time.strftime("%Y%m%d_%H%M%S")
	session_dir = os.path.join(base_dir, ts)
	os.makedirs(session_dir, exist_ok=True)
	return session_dir


def safe_dir_name(device_path):
	name = device_path.strip().replace("/", "_")
	return name or "device"


def ensure_device_dir(session_dir, device_path):
	device_dir = os.path.join(session_dir, safe_dir_name(device_path))
	os.makedirs(device_dir, exist_ok=True)
	return device_dir


if __name__ == "__main__":
	try:
		devices, capture_width, capture_height, capture_fps, alignment_cfg, crop_cfg = load_config(CONFIG_PATH)
	except Exception as e:
		print(f"Config error: {e}")
		raise SystemExit(1)

	manager = MultiUsbCamera(devices, width=capture_width, height=capture_height, fps=capture_fps)
	cameras = manager.start_all()
	if not cameras:
		print("No camera connected.")
		raise SystemExit(1)

	print("Preview started. Press 'a' to save images, 'x' to toggle recording, 'q' to quit.")

	base_output_dir = os.path.join(os.path.dirname(__file__), "output")
	os.makedirs(base_output_dir, exist_ok=True)

	session_dir = None
	img_counters = {device: 1 for device in devices}
	recording = False
	video_writers = {}
	last_processed = {}
	printed_sizes = set()

	try:
		while True:
			latest = manager.get_latest_frames()
			synced = {}

			if latest and all(device in latest for device in devices):
				base_time = latest[devices[0]][0]
				if all(abs(latest[device][0] - base_time) <= ALIGN_WINDOW_SEC for device in devices):
					for device in devices:
						frame = latest[device][1]
						synced[device] = (latest[device][0], frame)
					last_processed = preprocess_synced_frames(
						synced,
						alignment_cfg,
						crop_cfg,
						fallback_size=(capture_width, capture_height),
					)

			display = last_processed
			for device, (_, frame) in display.items():
				if device not in printed_sizes:
					h, w = frame.shape[:2]
					print(f"[{device}] frame size: {w}x{h}")
					printed_sizes.add(device)
				h, w = frame.shape[:2]
				preview = cv2.resize(frame, (max(1, w // 2), max(1, h // 2)))
				cv2.imshow(f"Camera - {device}", preview)

			if recording and last_processed:
				for device, (_, frame) in last_processed.items():
					if device in video_writers:
						video_writers[device].write(frame)

			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break
			if key == ord("a") and last_processed:
				if session_dir is None:
					session_dir = make_session_dir(base_output_dir)
				for device, (_, frame) in last_processed.items():
					device_dir = ensure_device_dir(session_dir, device)
					img_path = os.path.join(device_dir, f"{img_counters[device]}.png")
					cv2.imwrite(img_path, frame)
					img_counters[device] += 1
				print("Images saved.")
			if key == ord("x"):
				if not recording:
					if session_dir is None:
						session_dir = make_session_dir(base_output_dir)
					for device in devices:
						device_dir = ensure_device_dir(session_dir, device)
						video_path = os.path.join(device_dir, "record.mp4")
						if device in last_processed:
							h, w = last_processed[device][1].shape[:2]
						else:
							h = int(crop_cfg.get("height", DEFAULT_CROP_HEIGHT))
							w = int(crop_cfg.get("width", DEFAULT_CROP_WIDTH))
						fourcc = cv2.VideoWriter_fourcc(*"mp4v")
						video_writers[device] = cv2.VideoWriter(video_path, fourcc, capture_fps, (w, h))
					recording = True
					print("Recording started.")
				else:
					for writer in video_writers.values():
						writer.release()
					video_writers = {}
					recording = False
					print("Recording stopped.")

	finally:
		for writer in video_writers.values():
			writer.release()
		manager.stop_all()
		cv2.destroyAllWindows()
