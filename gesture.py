import cv2
import mediapipe as mp
#import handtrackingmodule as htm
import pyautogui
import threading
import time
import numpy as np 
import math
#from record import ScreenRecorder
#from cuong import record_vid


###################### SET UP ######################################
width_cam, height_cam = 640, 480
mp_hand = mp.solutions.hands.Hands(False, 2, 1, 0.75, 0.75)
cap = cv2.VideoCapture(0)
hand_status = ''
prev_time = 0

###################################################################

def key_press(direction):
    pyautogui.press(direction)



def navigate_slide(frame, x_mid, y_mid):

    if int(x_mid) > 440 : #and z_mean <= 0.021:
        hand_status = 'RIGHT'
        threading.Thread(target=key_press, args=('right',)).start()
        time.sleep(1)
    elif int(x_mid) < 200 : #and z_mean <= 0.0245:
        hand_status = 'LEFT'
        threading.Thread(target=key_press, args=('left',)).start()
        time.sleep(1)
    else:
        hand_status = 'NONE'
    cv2.putText(frame, hand_status, (430, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 250, 154), 2)

out = None
record_flag = False
recording_thread = None

# Function to start/stop recording
def record_vid():
    global out, record_flag
    screen_size = pyautogui.size()
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = 20.0
    out = cv2.VideoWriter('screen_recording.avi', fourcc, fps, screen_size)

    print("Recording started...")

    while record_flag:  # Ensure the flag is checked continuously
        img = pyautogui.screenshot()
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)

        time.sleep(0.05)  # Add a small delay for smoother recording

    # Release the recording when the flag becomes False
    out.release()
    print("Recording stopped.")

# Function to handle threading for starting/stopping recording
def manage_recording():
    global record_flag, recording_thread

    if not record_flag:
        # Start the recording in a new thread
        record_flag = True
        recording_thread = threading.Thread(target=record_vid)
        recording_thread.start()

    else:
        # Stop the recording
        record_flag = False
        if recording_thread is not None:
            recording_thread.join()



######################### MAIN PROCESS ################################
while True:
    success, frame = cap.read()
    frame_shape = frame.shape
    if not success:
        break 

    frame = cv2.flip(frame, 1)
    
    # if frame_shape is None:
    #     frame_shape = frame.shape
    
    frame.flags.writeable = False 
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    result = mp_hand.process(rgb_frame)



    frame.flags.writeable = True
    left_coordinate = None
    right_coordinate = None  
    left_hand_detected, right_hand_detected = False, False
    x_center, y_center = None, None
    


    if result.multi_hand_landmarks:
        for handlm, handedness in  zip(result.multi_hand_landmarks, result.multi_handedness):
            hand_label = handedness.classification[0].label
            h, w, _ = frame_shape

            cx, cy = int(handlm.landmark[8].x * w), int(handlm.landmark[8].y * h)
            if hand_label == "Left":
                left_hand_detected = True
                left_coordinate = (cx, cy)
            else: 
                right_hand_detected = True
                right_coordinate = (cx, cy)

            #----------------- Draw -----------------------#
            #cv2.circle(frame, (int(x_center), int(y_center)), 10, (255, 255, 0), -1) 
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)

            # Hiển thị tọa độ ngón trỏ trên frame
            if left_hand_detected:
                cv2.putText(frame, f"Left Index: {left_coordinate}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if right_hand_detected:
                cv2.putText(frame, f"Right Index: {right_coordinate}", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)      
                  

            #---------check condition for collision of 2 fingure--------#
            # if left_coordinate and right_coordinate:
            if left_hand_detected and right_hand_detected :
                #recorder = ScreenRecorder()
                #### checking
                len_line = math.hypot(left_coordinate[0]-right_coordinate[0], left_coordinate[1]-right_coordinate[1])
                if len_line < 24 and record_flag == False:
                    x_center, y_center = (left_coordinate[0]+right_coordinate[0])//2, (left_coordinate[1]+right_coordinate[1])//2    
                    record_flag = True
                    #record_vid(record_flag)
                    manage_recording()
                    time.sleep(4)
                elif len_line < 24 and record_flag == True:
                    record_flag = False
                    #record_vid(record_flag)
                    manage_recording()
                    #recorder.stop_recording()
                    time.sleep(4)
            
            # if right_hand_detected:
            #     x2, y2 = handlm.landmark[4].x * w, handlm.landmark[4].y * h    # Thumb
            #     x3, y3 = handlm.landmark[8].x * w, handlm.landmark[8].y * h    # Index finger

            #     # Draw circles on the thumb and index finger
            #     cv2.circle(frame, (int(x2), int(y2)), 10, (255, 0, 255), -1)
            #     cv2.circle(frame, (int(x3), int(y3)), 10, (255, 0, 255), -1)

            #     mid = math.hypot(x3 - x2, y3 - y2)
            #     if mid < 24:
            #         x_mid, y_mid = (x2 + x3) // 2, (y2 + y3) // 2
            #         cv2.circle(frame, (int(x_mid), int(y_mid)), 10, (255, 255, 0), -1)
            #         navigate_slide(frame, x_mid, y_mid)








                      #   function for start and stop recording
                    
                    ### to be continue

           

            

    #-------- FPS --------#
    curr_time = time.time()
    fps = 1 / (curr_time- prev_time)
    prev_time = curr_time

    cv2.putText(frame, f'FPS: {int(fps)}', (430,35), cv2.FONT_HERSHEY_COMPLEX,
              1, (255,0,255), 2)
    cv2.imshow("Hand Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
