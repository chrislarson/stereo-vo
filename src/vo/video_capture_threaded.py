import threading

import cv2


class VideoCaptureThreaded:
    def __init__(self, src_left: int, src_right: int, left_dim: (int, int), right_dim: (int, int)):
        self.thread = None
        self.src_left = src_left
        self.src_right = src_right
        self.cap_left = cv2.VideoCapture(self.src_left)
        self.cap_right = cv2.VideoCapture(self.src_right)
        self.cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, left_dim[0])
        self.cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, left_dim[1])
        self.cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, right_dim[0])
        self.cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, right_dim[1])
        self.grabbed_left, self.frame_left = self.cap_left.read()
        self.grabbed_right, self.frame_right = self.cap_right.read()
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.cap_left.set(var1, var2)
        self.cap_right.set(var1, var2)

    def start(self):
        if self.started:
            print("[!] Threaded video capturing has already been started.")
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed_left = self.cap_left.grab()
            grabbed_right = self.cap_right.grab()
            if not grabbed_left or not grabbed_right:
                continue
            with self.read_lock:
                self.grabbed_left = grabbed_left
                self.grabbed_right = grabbed_right

    def read(self):
        with self.read_lock:
            got_left, frame_left = self.cap_left.retrieve()
            got_right, frame_right = self.cap_right.retrieve()
            self.frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
            self.frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        return frame_left, frame_right

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap_left.release()
        self.cap_right.release()
