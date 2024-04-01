import cv2
import threading


class CameraFeed:
    def __init__(self):
        self.cap = None
        self.thread = None
        self.is_running = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()

    def start_feed(self):
        """Start the camera feed"""
        try:
            print("Starting webcam feed...")
            self.is_running = True
            if self.cap is None:
                self.cap = cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)
            self.thread = threading.Thread(target=self._capture_frames)
            self.thread.start()
            return 0
        except Exception as e:
            print(f"Error starting webcam feed: {e}")
            return -1

    def _capture_frames(self):
        """Internal method to continuously capture frames"""
        while self.is_running:
            # print("Capturing frame...")
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Couldn't capture frame from webcam.")
                break
            with self.frame_lock:
                self.latest_frame = frame

    def get_latest_frame(self):
        """Get the latest captured frame"""
        with self.frame_lock:
            return self.latest_frame

    def stop_feed(self):
        """Stop the camera feed"""
        try:
            self.is_running = False
            if self.thread:
                self.thread.join()
            if self.cap:
                self.cap.release()
            return 0
        except Exception as e:
            print(f"Error stopping webcam feed: {e}")
            return -1


if __name__ == "__main__":
    camera = CameraFeed()
    camera.start_feed()
    while True:
        frame = camera.get_latest_frame()
        if frame is not None:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    camera.stop_feed()
    cv2.destroyAllWindows()
