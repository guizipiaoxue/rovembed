import argparse
import asyncio
import base64
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import websockets

from dehaze.dehaze_uwcnn import UWCNNEnhancer

from camera.camera import MultiUsbCamera

from dehaze.dehaze_UDCP import CONFIG_PATH, FfmpegCamera, UDCPDehazer, load_camera_config, resize_for_preview


DEFAULT_HTTP_HOST = "0.0.0.0"
DEFAULT_WS_HOST = "0.0.0.0"
DEFAULT_PUBLIC_HOST = "192.168.127.10"
DEFAULT_HTTP_PORT = 8000
DEFAULT_WS_PORT = 8765
DEFAULT_CAPTURE_WIDTH = 640
DEFAULT_STREAM_FPS = 10.0
DEFAULT_JPEG_QUALITY = 80


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Camera Dehaze Display</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      background: #111;
      color: #f4f4f4;
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      background: #151515;
    }
    header {
      height: 52px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      border-bottom: 1px solid #303030;
      background: #1f1f1f;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 600;
    }
    #status {
      font-size: 14px;
      color: #b8b8b8;
    }
    main {
      flex: 1;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      padding: 12px;
      min-height: 0;
    }
    section {
      min-width: 0;
      min-height: 0;
      display: flex;
      flex-direction: column;
      border: 1px solid #333;
      background: #080808;
    }
    h2 {
      margin: 0;
      padding: 10px 12px;
      font-size: 16px;
      font-weight: 600;
      border-bottom: 1px solid #333;
      background: #202020;
    }
    .image-wrap {
      flex: 1;
      min-height: 0;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    img {
      display: block;
      max-width: 100%;
      max-height: calc(100vh - 128px);
      object-fit: contain;
    }
    @media (max-width: 900px) {
      main {
        grid-template-columns: 1fr;
      }
      img {
        max-height: 42vh;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>Camera Display</h1>
    <div id="status">连接中...</div>
  </header>
  <main>
    <section>
      <h2>原图</h2>
      <div class="image-wrap"><img id="original" alt="Original frame"></div>
    </section>
    <section>
      <h2>UDCP + UWCNN 去雾后</h2>
      <div class="image-wrap"><img id="dehazed" alt="Dehazed frame"></div>
    </section>
  </main>
  <script>
    const statusEl = document.getElementById("status");
    const originalEl = document.getElementById("original");
    const dehazedEl = document.getElementById("dehazed");
    const wsUrl = `ws://${location.hostname}:__WS_PORT__`;
    let socket = null;
    let frameCount = 0;
    let lastFpsTime = performance.now();
    let browserFps = 0;

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function connect() {
      socket = new WebSocket(wsUrl);
      socket.onopen = () => setStatus("已连接，等待画面...");
      socket.onclose = () => {
        setStatus("连接断开，正在重连...");
        setTimeout(connect, 1000);
      };
      socket.onerror = () => socket.close();
      socket.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type !== "frame") {
          return;
        }

        originalEl.src = `data:image/jpeg;base64,${data.original_jpeg}`;
        dehazedEl.src = `data:image/jpeg;base64,${data.dehazed_jpeg}`;
        frameCount += 1;

        const now = performance.now();
        if (now - lastFpsTime >= 1000) {
          browserFps = frameCount * 1000 / (now - lastFpsTime);
          frameCount = 0;
          lastFpsTime = now;
        }

        setStatus(
          `${data.width}x${data.height} | 发送 ${data.server_fps.toFixed(1)} fps | 浏览器 ${browserFps.toFixed(1)} fps`
        );
      };
    }

    connect();
  </script>
</body>
</html>
"""


class FrameStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.seq = 0
        self.payload = None

    def set_payload(self, payload):
        with self.lock:
            self.seq += 1
            payload["seq"] = self.seq
            self.payload = payload

    def get_payload(self):
        with self.lock:
            if self.payload is None:
                return self.seq, None
            return self.seq, dict(self.payload)


class DisplayHttpHandler(BaseHTTPRequestHandler):
    ws_port = DEFAULT_WS_PORT

    def do_GET(self):
        self._send_response(include_body=True)

    def do_HEAD(self):
        self._send_response(include_body=False)

    def _send_response(self, include_body):
        if self.path in ("/", "/index.html"):
            html = INDEX_HTML.replace("__WS_PORT__", str(self.ws_port)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            if include_body:
                self.wfile.write(html)
            return

        if self.path == "/health":
            body = b"ok\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if include_body:
                self.wfile.write(body)
            return

        self.send_error(404)

    def log_message(self, fmt, *args):
        print(f"[http] {self.address_string()} {fmt % args}")


class FrameProducer(threading.Thread):
    def __init__(self, args, store):
        super().__init__(daemon=True)
        self.args = args
        self.store = store
        self.running = threading.Event()
        self.running.set()
        self.manager = None
        self.ffmpeg_camera = None
        self.udcp_dehazer = UDCPDehazer(
            radius=args.radius,
            omega=args.omega,
            min_transmission=args.min_transmission,
            blur_size=args.blur_size,
            top_percent=args.top_percent,
        )
        self.uwcnn_enhancer = UWCNNEnhancer(
            weights_path=args.uwcnn_weights,
            device=args.uwcnn_device,
        )

    def stop(self):
        self.running.clear()

    def _open_capture(self):
        if self.args.backend == "ffmpeg":
            self.ffmpeg_camera = FfmpegCamera(
                self.args.device,
                self.args.width,
                self.args.height,
                self.args.capture_fps,
            )
            self.ffmpeg_camera.start()
            return

        self.manager = MultiUsbCamera(
            [self.args.device],
            width=self.args.width,
            height=self.args.height,
            fps=self.args.capture_fps,
        )
        cameras = self.manager.start_all()
        if not cameras:
            raise RuntimeError("No camera connected.")

    def _read_frame(self, last_capture_time):
        if self.ffmpeg_camera is not None:
            ok, frame = self.ffmpeg_camera.read()
            if not ok:
                return last_capture_time, None
            return time.time(), frame

        latest = self.manager.get_latest_frames()
        if self.args.device not in latest:
            time.sleep(0.01)
            return last_capture_time, None

        capture_time, frame = latest[self.args.device]
        if capture_time == last_capture_time:
            time.sleep(0.001)
            return last_capture_time, None
        return capture_time, frame

    def _close_capture(self):
        if self.manager is not None:
            self.manager.stop_all()
            self.manager = None
        if self.ffmpeg_camera is not None:
            self.ffmpeg_camera.stop()
            self.ffmpeg_camera = None

    def run(self):
        try:
            self._open_capture()
            self._produce_frames()
        except Exception as exc:
            print(f"[producer] stopped: {exc}")
        finally:
            self._close_capture()

    def _produce_frames(self):
        min_interval = 1.0 / max(self.args.stream_fps, 0.1)
        last_sent = 0.0
        last_capture_time = 0.0
        fps_time = time.time()
        fps_frames = 0
        server_fps = 0.0

        while self.running.is_set():
            capture_time, frame = self._read_frame(last_capture_time)
            if frame is None:
                continue
            last_capture_time = capture_time

            now = time.time()
            if now - last_sent < min_interval:
                continue
            last_sent = now

            original = resize_for_preview(frame, self.args.process_width)
            udcp_frame = self.udcp_dehazer.dehaze(original)
            dehazed = self.uwcnn_enhancer.enhance_bgr_frame(udcp_frame)

            original_jpeg = encode_jpeg_base64(original, self.args.jpeg_quality)
            dehazed_jpeg = encode_jpeg_base64(dehazed, self.args.jpeg_quality)
            if original_jpeg is None or dehazed_jpeg is None:
                continue

            fps_frames += 1
            if now - fps_time >= 1.0:
                server_fps = fps_frames / (now - fps_time)
                fps_frames = 0
                fps_time = now

            h, w = original.shape[:2]
            self.store.set_payload(
                {
                    "type": "frame",
                    "timestamp": capture_time,
                    "width": w,
                    "height": h,
                    "server_fps": server_fps,
                    "original_jpeg": original_jpeg,
                    "dehazed_jpeg": dehazed_jpeg,
                }
            )


def encode_jpeg_base64(frame, quality):
    quality = max(1, min(int(quality), 100))
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return None
    return base64.b64encode(encoded).decode("ascii")


def start_http_server(host, port, ws_port):
    handler = type("ConfiguredDisplayHttpHandler", (DisplayHttpHandler,), {"ws_port": ws_port})
    httpd = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def default_capture_size(cfg):
    cfg_w = max(1, int(cfg["width"]))
    cfg_h = max(1, int(cfg["height"]))
    width = min(cfg_w, DEFAULT_CAPTURE_WIDTH)
    height = max(1, int(round(width * cfg_h / cfg_w)))
    return width, height


async def websocket_handler(websocket, store):
    remote = websocket.remote_address
    print(f"[ws] client connected: {remote}")
    last_seq = 0
    try:
        while True:
            seq, payload = store.get_payload()
            if payload is not None and seq != last_seq:
                await websocket.send(json.dumps(payload, separators=(",", ":")))
                last_seq = seq
            await asyncio.sleep(0.005)
    except websockets.ConnectionClosed:
        pass
    finally:
        print(f"[ws] client disconnected: {remote}")


def parse_args():
    cfg = load_camera_config(CONFIG_PATH)
    default_width, default_height = default_capture_size(cfg)
    parser = argparse.ArgumentParser(description="Stream original and UDCP+UWCNN-dehazed camera frames over WebSocket.")
    parser.add_argument("--device", default=cfg["device"], help="Camera device path.")
    parser.add_argument("--width", type=int, default=default_width, help="Capture width.")
    parser.add_argument("--height", type=int, default=default_height, help="Capture height.")
    parser.add_argument("--capture-fps", type=int, default=int(cfg["fps"]), help="Camera capture FPS.")
    parser.add_argument("--stream-fps", type=float, default=DEFAULT_STREAM_FPS, help="WebSocket push FPS.")
    parser.add_argument("--backend", choices=("ffmpeg", "opencv"), default="ffmpeg", help="Video capture backend.")
    parser.add_argument("--process-width", type=int, default=320, help="Resize before dehaze and streaming; 0 keeps capture size.")
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY, help="JPEG quality from 1 to 100.")
    parser.add_argument("--radius", type=int, default=5, help="UDCP dark-channel radius.")
    parser.add_argument("--omega", type=float, default=0.95, help="UDCP haze removal strength.")
    parser.add_argument("--min-transmission", type=float, default=0.12, help="Lower bound for transmission.")
    parser.add_argument("--blur-size", type=int, default=7, help="Gaussian blur size for transmission refinement.")
    parser.add_argument("--top-percent", type=float, default=0.0, help="Atmospheric-light top percent; 0 uses fastest max pixel.")
    parser.add_argument("--uwcnn-weights", default=None, help="Path to a converted PyTorch .pth checkpoint.")
    parser.add_argument("--uwcnn-device", default=None, help="UWCNN device, for example cpu or cuda. Defaults automatically.")
    parser.add_argument("--http-host", default=DEFAULT_HTTP_HOST, help="HTTP bind host.")
    parser.add_argument("--http-port", type=int, default=DEFAULT_HTTP_PORT, help="HTTP port for browser page.")
    parser.add_argument("--ws-host", default=DEFAULT_WS_HOST, help="WebSocket bind host.")
    parser.add_argument("--ws-port", type=int, default=DEFAULT_WS_PORT, help="WebSocket port for frame data.")
    parser.add_argument("--public-host", default=DEFAULT_PUBLIC_HOST, help="Host/IP printed for remote browser access.")
    return parser.parse_args()


async def main_async(args):
    store = FrameStore()
    producer = FrameProducer(args, store)
    httpd = start_http_server(args.http_host, args.http_port, args.ws_port)
    producer.start()

    print(f"HTTP page: http://{args.public_host}:{args.http_port}")
    print(f"WebSocket: ws://{args.public_host}:{args.ws_port}")
    print("Open the HTTP page from 192.168.127.1. Press Ctrl+C here to stop.")

    try:
        async with websockets.serve(
            lambda websocket: websocket_handler(websocket, store),
            args.ws_host,
            args.ws_port,
            max_size=None,
            compression=None,
        ):
            await asyncio.Future()
    finally:
        producer.stop()
        producer.join(timeout=2.0)
        httpd.shutdown()
        httpd.server_close()


def main():
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
