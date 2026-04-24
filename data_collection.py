import cv2
import mediapipe as mp
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280) # Width
cap.set(4, 720)  # Height

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

while True:
    # 1. Get image from webcam
    success, img = cap.read()
    if not success: break
    
    # Flip image (mirror view)
    img = cv2.flip(img, 1)

    # 2. Convert to RGB (MediaPipe requires RGB, OpenCV uses BGR)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. Process the image to find hands
    results = hands.process(imgRGB)

    # 4. If hands are found
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Draw the skeleton on the hand
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Get coordinates of Index Finger Tip (ID 8)
            # handLms.landmark is a list of 21 points (0-20)
            # x and y are normalized (0.0 to 1.0), so we multiply by width/height
            h, w, c = img.shape
            cx, cy = int(handLms.landmark[8].x * w), int(handLms.landmark[8].y * h)

            # Draw the purple circle
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            
            # Print coordinates
            print(f"Index Tip: {cx}, {cy}")

    # Show the image
    cv2.imshow("Image", img)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()