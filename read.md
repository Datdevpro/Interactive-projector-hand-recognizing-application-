# def record_vid(record_flag):
#     global out

#     if record_flag:
#         # Start screen recording
#         screen_size = pyautogui.size()
#         fourcc = cv2.VideoWriter_fourcc(*"XVID")
#         fps = 20.0
#         out = cv2.VideoWriter('screen_recording.avi', fourcc, fps, screen_size)

#         print("Recording started...")

#         # Record screen until 'record_flag' is turned off
#         while record_flag:
#             img = pyautogui.screenshot()
#             frame = np.array(img)
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             out.write(frame)

#             # Simulate checking of flag again if needed
#             if not record_flag:
#                 break

#             # Add a small delay to mimic recording (remove in actual use)
#             time.sleep(0.05)

#     else:
#         # Stop screen recording
#         if out is not None:
#             out.release()
#             out = None
#             print("Recording stopped.")
