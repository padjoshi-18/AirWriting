import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import tensorflow as tf

# ---------------------------------------------------------
# 1. LOAD THE BRAIN
# ---------------------------------------------------------
print("Loading AI Model...")
model = tf.keras.models.load_model('air_writing_model.keras')
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
current_prediction = ""

# ---------------------------------------------------------
# 2. SETUP CAMERA & TRACKER
# ---------------------------------------------------------
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(detectionCon=0.8, maxHands=1)

# The Canvas and variables
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
px, py = 0, 0

print("System Ready! Show your hand to the camera.")

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1) 

    hands, img = detector.findHands(img, draw=True) 

    if hands:
        hand = hands[0]
        lmList = hand['lmList']
        x1, y1 = lmList[8][0], lmList[8][1]  # Index finger tip
        
        fingers = detector.fingersUp(hand)

        # --- LOGIC STATES ---

        # STATE 3: PREDICT & CLEAR (Index, Middle, and Ring fingers are up)
        if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:
            print("Processing Prediction...")
            imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
            
            if cv2.countNonZero(imgGray) > 0:
                coords = cv2.findNonZero(imgGray)
                x, y, w, h = cv2.boundingRect(coords)
                
                # 1. Crop exactly around the letter (tight crop)
                tight_crop = imgGray[y:y+h, x:x+w]
                
                # 2. Find the longest side
                max_dim = max(w, h)
                
                # 3. Calculate black space to add to center it perfectly
                top = (max_dim - h) // 2
                bottom = max_dim - h - top
                left = (max_dim - w) // 2
                right = max_dim - w - left
                
                # 4. Pad with black to make a perfect square
                square_img = cv2.copyMakeBorder(tight_crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
                
                # 5. Add a 20% outer margin
                margin = int(max_dim * 0.2)
                final_padded_img = cv2.copyMakeBorder(square_img, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=0)
                
                # --- THE EMNIST "CENTER OF MASS" TRICK ---
                # EMNIST letters are centered by their physical weight, not their borders.
                M = cv2.moments(final_padded_img)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    # Calculate how far to shift the image to put the weight in the dead center
                    shift_x = final_padded_img.shape[1]//2 - cX
                    shift_y = final_padded_img.shape[0]//2 - cY
                    
                    # Shift the image
                    translation_matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                    final_padded_img = cv2.warpAffine(final_padded_img, translation_matrix, (final_padded_img.shape[1], final_padded_img.shape[0]))
                # ----------------------------------------------

                # 6. Shrink down to 28x28 (INTER_AREA softens it perfectly)
                resized_img = cv2.resize(final_padded_img, (28, 28), interpolation=cv2.INTER_AREA)

                # --- DEBUGGING WINDOW ---
                # Scale the tiny 28x28 image up by 10x so our human eyes can see it
                debug_img = cv2.resize(resized_img, (280, 280), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("What the AI Sees", debug_img)
                # ------------------------

                # Format for the CNN
                input_img = resized_img.astype('float32') / 255.0
                input_img = np.reshape(input_img, (1, 28, 28, 1))
                
                # Predict
                prediction_array = model.predict(input_img, verbose=0)
                predicted_index = np.argmax(prediction_array)
                current_prediction = ALPHABET[predicted_index]
                print(f"I see a: {current_prediction}")
            
            # Clear the canvas immediately after predicting
            imgCanvas = np.zeros((720, 1280, 3), np.uint8)

        # STATE 1: HOVER MODE (Index + Middle Up, but Ring is DOWN)
        elif fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:
            px, py = x1, y1
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

        # STATE 2: WRITE MODE (Only Index Up, Middle is DOWN)
        elif fingers[1] == 1 and fingers[2] == 0:
            cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
            if px == 0 and py == 0:
                px, py = x1, y1
            
            # Draw on canvas (thickness 30 works beautifully with the new interpolation)
            cv2.line(imgCanvas, (px, py), (x1, y1), (255, 255, 255), 30)
            cv2.line(img, (px, py), (x1, y1), (0, 255, 0), 30)
            px, py = x1, y1

    else:
        px, py = 0, 0

    # Merge Canvas and Video
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # Display the Prediction on the Screen
    cv2.putText(img, f"Prediction: {current_prediction}", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    cv2.imshow("Air Writing", img)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()