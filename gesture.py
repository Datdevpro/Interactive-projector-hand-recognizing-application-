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
mp_hand = mp.solutions.hands.Hands(False, 1, 1, 0.75, 0.75) # (static_image_mode, max_num_hands, min_detection_confidence, min_tracking_confidence, model_complexity)
cap = cv2.VideoCapture(0)
hand_status = ''
prev_time = 0

# Add these global variables
record_flag = False
recording_thread = None
last_toggle_time = 0

###################################################################

def key_press(direction):
    pyautogui.press(direction)

def record_screen():
    global record_flag
    screen_size = pyautogui.size()
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = 30  # Increased from 20.0 to 30.0
    out = cv2.VideoWriter('screen_recording.avi', fourcc, fps, screen_size)

    print("Recording started...")
    start_time = time.time()
    frame_count = 0

    while record_flag:
        img = pyautogui.screenshot()
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.write(frame)
        frame_count += 1
        
        # Calculate the time elapsed
        elapsed_time = time.time() - start_time
        
        # Calculate the time we should have spent for the current frame count
        expected_time = frame_count / fps
        
        # If we're ahead of schedule, wait for a bit
        if elapsed_time < expected_time:
            time.sleep(expected_time - elapsed_time)

    out.release()
    print("Recording stopped and saved.")

def manage_recording():
    global record_flag, recording_thread, last_toggle_time

    current_time = time.time()
    if current_time - last_toggle_time < 1:  # Prevent rapid toggling
        return

    if not record_flag:
        record_flag = True
        recording_thread = threading.Thread(target=record_screen)
        recording_thread.start()
        print("Recording started")
    else:
        record_flag = False
        if recording_thread is not None:
            recording_thread.join()
        print("Recording stopped")

    last_toggle_time = current_time

def navigate_slide(frame, x, y):
    frame_height, frame_width = frame.shape[:2]
    if x < frame_width * 0.3:
        key_press('left')
        time.sleep(1.1)
    elif x > frame_width * 0.7:
        key_press('right')
        time.sleep(1.1)

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
            # if left_hand_detected and right_hand_detected :
            #     #recorder = ScreenRecorder()
            #     #### checking
            #     len_line = math.hypot(left_coordinate[0]-right_coordinate[0], left_coordinate[1]-right_coordinate[1])
            #     if len_line < 24 and record_flag == False:
            #         x_center, y_center = (left_coordinate[0]+right_coordinate[0])//2, (left_coordinate[1]+right_coordinate[1])//2    
            #         record_flag = True
            #         #record_vid(record_flag)
            #         #manage_recording()
            #         #time.sleep(4)
            #     elif len_line < 24 and record_flag == True:
            #         record_flag = False
                    #record_vid(record_flag)
                    #manage_recording()
                    #recorder.stop_recording()
                    #time.sleep(4)
            
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

            if hand_label == "Right":
                # Process right hand for slide navigation and recording control
                thumb_tip = handlm.landmark[4]
                index_tip = handlm.landmark[8]
                
                x_thumb, y_thumb = int(thumb_tip.x * w), int(thumb_tip.y * h)
                x_index, y_index = int(index_tip.x * w), int(index_tip.y * h)
                
                cv2.circle(frame, (x_thumb, y_thumb), 10, (255, 0, 0), -1)
                cv2.circle(frame, (x_index, y_index), 10, (0, 255, 0), -1)
                
                # Check for thumb and index finger collision
                distance = math.hypot(x_thumb - x_index, y_thumb - y_index)
                if distance < 24:
                    cv2.circle(frame, ((x_thumb + x_index) // 2, (y_thumb + y_index) // 2), 10, (0, 255, 255), -1)
                    manage_recording()  # This will toggle recording on/off
                else:
                    # If not in collision, use for slide navigation
                    navigate_slide(frame, x_index, y_index)

    # Display recording status on the frame
    cv2.putText(frame, f"Recording: {'ON' if record_flag else 'OFF'}", (10, 110), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if record_flag else (0, 0, 255), 2)

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

# Ensure recording is stopped when exiting the program
if record_flag:
    record_flag = False
    if recording_thread is not None:
        recording_thread.join()

cap.release()
cv2.destroyAllWindows()
