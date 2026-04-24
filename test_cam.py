import cv2

# Open the default camera (0 is usually the built-in webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit the window.")

while True:
    # Read a frame from the webcam
    success, frame = cap.read()
    
    if not success:
        print("Failed to read frame.")
        break

    # Display the frame in a window named "Air Writing Test"
    cv2.imshow("Air Writing Test", frame)

    # Wait 1ms for a key press; if 'q' is pressed, break loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()