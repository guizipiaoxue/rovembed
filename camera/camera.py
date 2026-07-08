import sys
import os
import time
import cv2
import threading


DEFAULT_V4L_CAMERA_DEVICES = [
    "/dev/v4l/by-path/platform-xhci-hcd.2.auto-usb-0:1.1:1.0-video-index0",
    "/dev/v4l/by-path/platform-xhci-hcd.2.auto-usb-0:1.2:1.0-video-index0",
]


class UsbCamera(threading.Thread):
    def __init__(self, device_path, width=1920, height=1080, fps=30):
        super().__init__()
        self.device_path = device_path
        self.capture_path = self._normalize_device_path(device_path)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.cap = None
        self.running = False
        self.daemon = True

        # 缓存存放时间戳和画面
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_time = 0.0

        if not self._connect():
            raise Exception(f"无法连接到USB相机: {device_path}")

    @staticmethod
    def _normalize_device_path(device_path):
        if os.path.exists(device_path):
            return device_path

        v4l_path = os.path.join("/dev/v4l", device_path.lstrip("/"))
        if os.path.exists(v4l_path):
            return v4l_path

        return device_path

    def _connect(self):
        self.cap = cv2.VideoCapture(self.capture_path, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            print(f"[{self.device_path}] 打开设备失败: {self.capture_path}")
            return False

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        actual_codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
        real_path = os.path.realpath(self.capture_path)
        print(
            f"[{self.device_path}] 请求: {self.width}x{self.height}@{self.fps} "
            f"实际: {actual_w}x{actual_h}@{actual_fps:.2f} codec={actual_codec} "
            f"v4l={self.capture_path} real={real_path}"
        )
        return True

    def run(self):
        self.running = True
        while self.running:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                capture_time = time.time()
                with self.lock:
                    self.latest_frame = frame
                    self.latest_time = capture_time
            else:
                time.sleep(0.01)

    def stop(self):
        print(f"[{self.device_path}] 正在停止取流和断开连接...")
        self.running = False
        self.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()

    def get_latest_data(self):
        """返回缓存的(时间戳, 当前帧)"""
        with self.lock:
            return self.latest_time, self.latest_frame


class MultiUsbCamera:
    """多相机管理器：每个相机一个线程，避免阻塞。"""

    def __init__(self, device_paths, width=1920, height=1080, fps=30):
        self.device_paths = list(device_paths)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.cameras = []

    def start_all(self):
        self.cameras = []
        for device_path in self.device_paths:
            try:
                cam_worker = UsbCamera(
                    device_path,
                    width=self.width,
                    height=self.height,
                    fps=self.fps,
                )
                cam_worker.start()
                self.cameras.append(cam_worker)
            except Exception as e:
                print(e)
        return self.cameras

    def stop_all(self):
        for cam_worker in self.cameras:
            cam_worker.stop()

    def get_latest_frames(self):
        """返回 {device: (timestamp, frame)}"""
        data = {}
        for cam_worker in self.cameras:
            timestamp, frame = cam_worker.get_latest_data()
            if frame is not None:
                data[cam_worker.device_path] = (timestamp, frame)
        return data


if __name__ == "__main__":
    CAMERA_DEVICES = DEFAULT_V4L_CAMERA_DEVICES

    manager = MultiUsbCamera(CAMERA_DEVICES)
    cameras = manager.start_all()

    if not cameras:
        print("未成功连接任何相机，退出...")
        sys.exit(1)

    print("进入多相机视频流预览（按 'q' 键退出预览）...")

    try:
        while True:
            display_frames = {}
            base_time = 0.0

            latest = manager.get_latest_frames()
            if latest:
                first_dev = list(latest.keys())[0]
                base_time, base_frame = latest[first_dev]
                display_frames[first_dev] = base_frame

            for device, (timestamp, frame) in latest.items():
                if device == first_dev:
                    continue
                if abs(timestamp - base_time) < 0.035:
                    display_frames[device] = frame
                else:
                    print(
                        f"警告：[{device}] 当前读取帧未能在时间窗内与基准对齐。"
                        f"差值: {abs(timestamp - base_time):.3f}s"
                    )

            for device, frame in display_frames.items():
                h, w = frame.shape[:2]
                resized = cv2.resize(frame, (w // 2, h // 2))
                cv2.imshow(f"Camera - {device}", resized)

            if cv2.waitKey(30) & 0xFF == ord("q"):
                break

    finally:
        manager.stop_all()
        cv2.destroyAllWindows()
        print("所有相机已断开连接，程序退出。")
