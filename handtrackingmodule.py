import cv2
import mediapipe as mp
import time
import math

############ LƯU Ý FILE NÀY CÓ HÀM ĐƯỢC VIẾT CHỈ DÙNG CHO VÀI ĐIỂM TRÊN TAY CHỨ KHÔNG GENERAL ###################

class handDetector():
    def __init__(self, mode=False, maxHands=2, min_detection_conf=0.5, min_tracking_conf=0.5):
        self.mode = mode   ### detect static image or stream video
        self.maxHands = maxHands
        self.min_detection_conf = min_detection_conf
        self.min_tracking_conf = min_tracking_conf

        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,
                                        self.maxHands,    
                                    )
        self.mpDraw = mp.solutions.drawing_utils
        self.landmark_drawing_specification = self.mpDraw.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)  # Green color
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, 
                                               handLms,
                                               self.mpHands.HAND_CONNECTIONS, 
                                            landmark_drawing_spec=self.landmark_drawing_specification)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        # xList = []
        # yList = []
        # bbox = []
        self.landmart_list = []
        #self.results = self.hands.process(self.imgRGB)
        # print(self.results.multi_hand_landmarks)
        # print(type(self.results.multi_hand_landmarks))
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo] ### chọn cái tay đầu tiên trong danh sách
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # xList.append(cx)
                # yList.append(cy)
                # print(id, cx, cy)
                self.landmart_list.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        #     xmin, xmax = min(xList), max(xList)
        #     ymin, ymax = min(yList), max(yList)
        #     bbox = xmin, ymin, xmax, ymax

        # if draw:
        #     cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
        # (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)
        
        return self.landmart_list #, bbox

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.landmart_list[self.tipIds[0]][1] > self.landmart_list[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 Fingers
        for id in range(1, 5):
            if self.landmart_list[self.tipIds[id]][2] < self.landmart_list[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True):

        x1, y1 = self.landmart_list[p1][1], self.landmart_list[p1][2]
        x2, y2 = self.landmart_list[p2][1], self.landmart_list[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        if draw:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        
        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]

    def main():
        pTime = 0
        cap = cv2.VideoCapture(1)
        detector = handDetector()
        while True:
            success, img = cap.read()
            img = detector.findHands(img)
            landmart_list = detector.findPosition(img)
            if len(landmart_list) != 0:
                print(landmart_list[4])
        
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            
            cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
            (255, 0, 255), 3)
            
            cv2.imshow("Image", img)
            cv2.waitKey(1)

# if __name__ == "__main__":
#     main()