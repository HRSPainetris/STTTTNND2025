import os
import numpy as np

# Import Image Processing library
import cv2
import imutils
from imutils.video import VideoStream
from imutils import face_utils
import dlib

# Time library
import time
from time import strftime
from datetime import datetime

# Audio playing library
from pygame import mixer
mixer.init()

# Grab path to current working directory
CWD_PATH = os.getcwd()
audio_path = os.path.join(CWD_PATH,"audio_files")
sys_img_path = os.path.join(CWD_PATH, "system_images")

###################################################
##                     AUDIO                     ##
###################################################

## Phat loa
def phat_loa(audio_file_name):
    audio_file_name = os.path.join(audio_path, audio_file_name)
    if mixer.music.get_busy():
        speaker_wait = 1
        while speaker_wait == 1:
            if not mixer.music.get_busy():
                mixer.music.load(audio_file_name)
                mixer.music.play()
                speaker_wait = 0
    elif not mixer.music.get_busy():
        mixer.music.load(audio_file_name)
        mixer.music.play()
        
## Phat loa va cho den ket thuc
def phat_loa_until_end(audio_file_name):
    audio_file_name = os.path.join(audio_path, audio_file_name)
    if mixer.music.get_busy():
        speaker_wait = 1
        while speaker_wait == 1:
            if not mixer.music.get_busy():
                mixer.music.load(audio_file_name)
                mixer.music.play()
                speaker_wait = 0
                speaking = 1 
                while speaking == 1:
                    if not mixer.music.get_busy():
                        speaking=0            
                        time.sleep(0.05)
    elif not mixer.music.get_busy():
        mixer.music.load(audio_file_name)
        mixer.music.play()
        speaking = 1
        while speaking==1:
            if not mixer.music.get_busy():
                speaking=0            
            time.sleep(0.05)
            
###################################################
##                VISUALIZATION                  ##
###################################################
## SIZE OF IMAGE
w_process_size = 320
h_process_size = 240  

w_show_size = 320
h_show_size = 240

system_name_img_width = w_show_size

## LOCATION OF IMAGE
system_name_location = [0, 0]

left_show_location = [0, 240]
right_show_location = [0, 485]
driver_show_location = [0, 760]

info_show_location = [480, 450]
info_img_width = 960

## FUNCTION: show image in a fixed window
def showInFixedWindow(winname, img, x, y):
    img = imutils.resize(img, width = w_show_size)
    cv2.namedWindow(winname, flags=cv2.WINDOW_GUI_NORMAL + cv2.WINDOW_AUTOSIZE)  # Create a named window
    cv2.moveWindow(winname, x, y)   # Move it to (x,y)
    cv2.imshow(winname, img)
    
# FUNCTION: Show the left camera image in a fixed window
def visualize_left(left_frame):
    showInFixedWindow("Left Camera", left_frame, left_show_location[0], left_show_location[1])

# FUNCTION: Show the right camera image in a fixed window
def visualize_right(right_frame):
    showInFixedWindow("Right Camera", right_frame, right_show_location[0], right_show_location[1])
    
# FUNCTION: Show the driver camera image in a fixed window
def visualize_driver(driver_frame):
    showInFixedWindow("Driver Camera", driver_frame, driver_show_location[0], driver_show_location[1])
    
# FUNCTION: Show the system name image in a fixed window
def visualize_system_name():
    img = cv2.imread(os.path.join(sys_img_path, "KHKT2022_NDC.png"))
    img = imutils.resize(img, width=system_name_img_width)
    (h_sysname_img, w_sysname_img) = img.shape[:2]
    print("h_sysname_img: ",h_sysname_img)
    print("w_sysname_img: ",w_sysname_img)
    showInFixedWindow('System Name', img, system_name_location[0], system_name_location[1])
    cv2.waitKey(1)

# FUNCTION: Show the notice image in a fixed window
def visualize_info(info_img_name):
    if cv2.getWindowProperty("System Notice", cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyWindow("System Notice")
    img = cv2.imread(os.path.join(sys_img_path, info_img_name + ".png"))
    img = imutils.resize(img, width=info_img_width)
    showInFixedWindow('System Notice', img, info_show_location[0], info_show_location[1])
    print("System Notice: ", info_img_name)
    cv2.waitKey(1)
            
###################################################
##               VISUALIZE & AUDIO               ##
###################################################
# 1. Chao mung va yeu cau nhap thong tin hai trinh
def phat_loa_1_chao_mung():
    # phat_loa("1_chao_mung.mp3")
    phat_loa_until_end("1_chao_mung.mp3")
    visualize_info("cam_on")
    print("1. Da phat loa chao mung va yeu cau nhap thong tin")
    time.sleep(0.5)
    
    
# 2. Da ghi nhan thong tin hai trinh
def phat_loa_2_da_ghi_thong_tin():
    # phat_loa("2_da_ghi_thong_tin.mp3")
    phat_loa_until_end("2_da_ghi_thong_tin.mp3")
    print("2. Da phat loa da ghi nhan thong tin")
    time.sleep(0.5)
    
# 3. Canh bao va cham ben tu 2 ben
def phat_loa_3_cham_2ben():
    # phat_loa("3_cham_2ben.mp3")
    phat_loa_until_end("3_cham_2ben.mp3")
    print("3. Da phat loa canh bao va cham ben tu 2 ben")
    time.sleep(0.5)

# 4. Canh bao va cham tu ben phai
def phat_loa_4_cham_phai():
    # phat_loa("4_cham_phai.mp3")
    phat_loa_until_end("4_cham_phai.mp3")
    print("4. Da phat loa canh bao va cham tu ben phai")
    time.sleep(0.5)

# 5. Canh bao va cham tu ben trai
def phat_loa_5_cham_trai():
    # phat_loa("5_cham_trai.mp3")
    phat_loa_until_end("5_cham_trai.mp3")
    print("5. Da phat loa canh bao va cham tu ben trai")
    time.sleep(0.5)
    
# 6. Canh bao phat hien nguoi o 2 ben
def phat_loa_6_nguoi_2ben():
    # phat_loa("6_nguoi_2ben.mp3")
    phat_loa_until_end("6_nguoi_2ben.mp3")
    print("6. Da phat loa canh bao phat hien nguoi o 2 ben")
    time.sleep(0.5)

# 7. Canh bao phat hien nguoi o ben phai
def phat_loa_7_nguoi_phai():
    # phat_loa("7_nguoi_phai.mp3")
    phat_loa_until_end("7_nguoi_phai.mp3")
    print("7. Da phat loa canh bao phat hien nguoi o ben phai")
    time.sleep(0.5)

# 8. Canh bao phat hien nguoi o ben trai
def phat_loa_8_nguoi_trai():
    # phat_loa("8_nguoi_trai.mp3")
    phat_loa_until_end("8_nguoi_trai.mp3")
    print("8. Da phat loa canh bao phat hien nguoi o ben trai")
    time.sleep(0.5)
    
# 9. Chu y bien bao o 2 ben
def phat_loa_9_bien_bao_2ben():
    # phat_loa("9_bien_bao_2ben.mp3")
    phat_loa_until_end("9_bien_bao_2ben.mp3")
    print("9. Da phat loa chu y bien bao o 2 ben")
    time.sleep(0.5)

# 10. Chu y bien bao o ben phai
def phat_loa_10_bien_bao_phai():
    # phat_loa("10_bien_bao_phai.mp3")
    phat_loa_until_end("10_bien_bao_phai.mp3")
    print("10. Da phat loa chu y bien bao o ben phai")
    time.sleep(0.5)
    
# 11. Chu y bien bao o ben trai
def phat_loa_11_bien_bao_trai():
    # phat_loa("11_bien_bao_trai.mp3")
    phat_loa_until_end("11_bien_bao_trai.mp3")
    print("11. Da phat loa chu y bien bao o ben trai")
    time.sleep(0.5)

# 12. Canh bao co cau o phia truoc
def phat_loa_12_cau_truoc():
    # phat_loa("12_cau_truoc.mp3")
    phat_loa_until_end("12_cau_truoc.mp3")
    print("12. Da phat loa canh bao co cau o phia truoc")
    time.sleep(0.5)
    
# 13. Canh bao co cau o ben trai
def phat_loa_13_cau_trai():
    # phat_loa("13_cau_trai.mp3")
    phat_loa_until_end("13_cau_trai.mp3")
    print("13. Da phat loa canh bao co cau o ben trai")
    time.sleep(0.5)

# 14. Canh bao co cau o ben phai
def phat_loa_14_cau_phai():
    # phat_loa("14_cau_phai.mp3")
    phat_loa_until_end("14_cau_phai.mp3")
    print("14. Da phat loa canh bao co cau o ben phai")
    time.sleep(0.5)
    
# 15. Canh bao chieu cao tau
def phat_loa_15_chieu_cao_tau():
    # phat_loa("15_chieu_cao_tau.mp3")
    phat_loa_until_end("15_chieu_cao_tau.mp3")
    print("15. Da phat loa canh bao chieu cao tau")
    time.sleep(0.5)

    

    
if __name__ == "__main__":
    phat_loa_1_chao_mung()
    # phat_loa_2_da_ghi_thong_tin()
    # phat_loa_3_cham_2ben()
    # phat_loa_4_cham_phai()
    # phat_loa_5_cham_trai()
    # phat_loa_6_nguoi_2ben()
    # phat_loa_7_nguoi_phai()
    # phat_loa_8_nguoi_trai()
    # phat_loa_9_bien_bao_2ben()
    # phat_loa_10_bien_bao_phai()
    # phat_loa_11_bien_bao_trai()
    # phat_loa_12_cau_truoc()
    # phat_loa_13_cau_trai()
    # phat_loa_14_cau_phai()
    # phat_loa_15_chieu_cao_tau()
    