import cv2
import time
import multiprocessing as mp

class Camera():
    
    def __init__(self,rtsp_url):        
        #load pipe for data transmittion to the process
        self.parent_conn, child_conn = mp.Pipe()
        #load process
        self.p = mp.Process(target=self.update, args=(child_conn,rtsp_url))        
        #start process
        self.p.daemon = True
        self.p.start()
        
    def end(self):
        #send closure request to process
        
        self.parent_conn.send(2)
        
    def update(self,conn,rtsp_url):
        #load cam into seperate process
        
        print("Cam Loading...")
        cap = cv2.VideoCapture(rtsp_url,cv2.CAP_FFMPEG)   
        print("Cam Loaded...")
        run = True
        
        while run:
            
            #grab frames from the buffer
            cap.grab()
            
            #recieve input data
            rec_dat = conn.recv()
            
            
            if rec_dat == 1:
                #if frame requested
                ret,frame = cap.read()
                conn.send(frame)
                
            elif rec_dat ==2:
                #if close requested
                cap.release()
                run = False
                
        print("Camera Connection Closed")        
        conn.close()
    
    def get_frame(self,resize=None):
        ###used to grab frames from the cam connection process
        
        ##[resize] param : % of size reduction or increase i.e 0.65 for 35% reduction  or 1.5 for a 50% increase
             
        #send request
        self.parent_conn.send(1)
        frame = self.parent_conn.recv()
        
        #reset request 
        self.parent_conn.send(0)
        
        #resize if needed
        if resize == None:            
            return frame
        else:
            return self.rescale_frame(frame,resize)
        
    def rescale_frame(self,frame, percent=65):
        
        return cv2.resize(frame,None,fx=percent,fy=percent) 
    
cam1_url = "rtsp://khkt2024cam1:khkt2024@ndc!@192.168.200.160:554/stream2"
cam1 = Camera(cam1_url)
cam2_url = "rtsp://khkt2024cam2:khkt2024@ndc!@192.168.200.189:554/stream2"
cam2 = Camera(cam2_url)

print(f"Camera 1 is alive?: {cam1.p.is_alive()}")
print(f"Camera 2 is alive?: {cam1.p.is_alive()}")

while(1):
    frame1 = cam1.get_frame(0.65)
    frame2 = cam2.get_frame(0.65)
    
    cv2.imshow("Cam1",frame1)
    cv2.imshow("Cam2",frame2)
    
    key = cv2.waitKey(1)

    if key == 13: #13 is the Enter Key
        break

cv2.destroyAllWindows()     

cam1.end()
cam2.end()
