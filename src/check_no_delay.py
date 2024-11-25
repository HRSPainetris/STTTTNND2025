import cv2
import time
import threading
import queue
import cv2, queue, threading, time
import numpy as np

# https://medium.com/@vikas.c20/optimizing-rtsp-video-processing-in-opencv-overcoming-fps-discrepancies-and-buffering-issues-463e204c7b86

# bufferless VideoCapture
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        self.lock = threading.Lock()
        self.running = True  # Flag to indicate if the thread should keep running
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # Discard previous frame
                except queue.Empty:
                    pass
            self.q.put(frame)
            self.state=ret

    def read(self):
        return self.q.get(),self.state

    def stop(self):
        self.running = False
        self.t.join()  # Wait for the thread to exit




def main():
    vs1 = VideoCapture("rtsp://khkt2024left:khkt2024@ndc!@192.168.0.101:554/stream1")
    vs2 = VideoCapture("rtsp://khkt2024right:khkt2024@ndc!@192.168.0.100:554/stream1")
    start_time2 = time.time()
    frame_rate = 30
    fps = 0
    # cv2.namedWindow("Live Stream", cv2.WINDOW_NORMAL)
    
    while True:
        frame_1,success = vs1.read()
        frame_2,success = vs2.read()
        img1 = cv2.resize(frame_1, (640, 360), interpolation = cv2.INTER_LINEAR)
        img2 = cv2.resize(frame_2, (640, 360), interpolation = cv2.INTER_LINEAR)
        start_time = time.time()
        if not success:
            break

        loop_time = time.time() - start_time
        delay = max(1, int((1 / frame_rate - loop_time) * 1000))
        key = cv2.waitKey(delay) & 0xFF

        if key == ord('q'):
            break

        loop_time2 = time.time() - start_time
        if loop_time2 > 0:
            fps = 0.9 * fps + 0.1 / loop_time2
            print(fps)
        cv2.putText(img1, f"FPS Cam 1: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        # cv2.imshow("Live Stream 1", frame_1)
        cv2.putText(img2, f"FPS Cam 2: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        # cv2.imshow("Live Stream 2", frame_2)
        Hori = np.concatenate((img1, img2), axis=1) 
        cv2.imshow('Cam1+2', Hori)
        # print("Cam 1: {} x {}. Cam 2: {} x {}.".format(frame_width1,frame_height1,frame_width2,frame_height2))

    
    total_time = time.time() - start_time2
    print("Total time taken:", total_time, "seconds")

    cv2.destroyAllWindows()
    vs1.cap.release()
    vs2.cap.release()

if __name__ == "__main__":
    main()