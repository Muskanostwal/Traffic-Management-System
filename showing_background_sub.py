import cv2

# Initialize video capture
cap = cv2.VideoCapture('video_vehicles.mp4')

# Background Subtractor
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for better visibility
    frame = cv2.resize(frame, (1020, 500))

    # Apply background subtraction
    mask = algo.apply(frame)

    # Show the black-and-white mask frame
    cv2.imshow('Black-and-White Frame', mask)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
