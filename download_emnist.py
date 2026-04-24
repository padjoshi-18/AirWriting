import cv2
import mediapipe as mp
import numpy as np
import os
import time

# --- CONFIGURATION ---
FOLDER_PATH = "Data"   # The main folder
TARGET_LETTER = "A"    # CHANGE THIS to 'B', 'C', etc. as you go!
CANVAS_SIZE = (300, 300) 
# ---------------------

# 1. Create the folder structure
save_path = os.path.join(FOLDER_PATH, TARGET_LETTER)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 2. Setup Camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280) # Width
cap.set(4, 720)  # Height

# 3. Setup MediaPipe (The "Eyes")
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False, 
                      max_num_hands=1, 
                      min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Variables for drawing
xp, yp = 0, 0
canvas = np.zeros((720, 1280, 3), np.uint8) # Black drawing board

print(f"--- COLLECTING DATA FOR: '{TARGET_LETTER}' ---")
print("Index Finger UP:  Draw")
print("Two Fingers UP:   Pause/Move (Lift Pen)")
print("Key 's':          SAVE Image")
print("Key 'c':          CLEAR Canvas")
print("Key 'q':          QUIT")

while True:
    # 1. Read Frame
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1) # Mirror view

    # 2. Find Hands
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Get coordinates
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            if len(lmList) != 0:
                # Tip of Index Finger (8) and Middle Finger (12)
                x1, y1 = lmList[8][1], lmList[8][2]
                x2, y2 = lmList[12][1], lmList[12][2]

                # Check which fingers are up
                fingers = []
                # Index (Tip above lower joint)
                if lmList[8][2] < lmList[6][2]: fingers.append(1)
                else: fingers.append(0)
                # Middle
                if lmList[12][2] < lmList[10][2]: fingers.append(1)
                else: fingers.append(0)

                # --- DRAW MODE (Only Index Up) ---
                if fingers[0] == 1 and fingers[1] == 0:
                    cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                    
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1

                    # Draw on canvas (White ink) and on screen (Purple ink)
                    cv2.line(canvas, (xp, yp), (x1, y1), (255, 255, 255), 25)
                    cv2.line(img, (xp, yp), (x1, y1), (255, 0, 255), 25)
                    
                    xp, yp = x1, y1

                # --- HOVER MODE (Index + Middle Up) ---
                elif fingers[0] == 1 and fingers[1] == 1:
                    xp, yp = 0, 0 # Reset drawing point

    # 3. Merge Canvas and Webcam
    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, canvas)

    # 4. Show Images
    cv2.imshow("Air Writing", img)
    # Also show the clean Black & White canvas (this is what the AI will see)
    cv2.imshow("AI View", cv2.resize(canvas, (400, 300)))

    # 5. Controls
    key = cv2.waitKey(1)
    
    if key == ord('c'):
        canvas = np.zeros((720, 1280, 3), np.uint8)
        print("Canvas Cleared")
        
    if key == ord('s'):
        # Resize to 28x28 (standard AI size) or 300x300 for storage
        img_save = cv2.resize(canvas, (300, 300))
        
        # Invert colors so it looks like black ink on white paper (standard for EMNIST)
        img_save = cv2.bitwise_not(img_save)
        
        filename = f"{save_path}/{int(time.time())}.jpg"
        cv2.imwrite(filename, img_save)
        print(f"SAVED: {filename}")
        canvas = np.zeros((720, 1280, 3), np.uint8) # Auto-clear

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()