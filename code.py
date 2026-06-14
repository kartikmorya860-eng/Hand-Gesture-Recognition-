import cv2
import numpy as np

# Step 1: Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    # Step 2: Read each frame
    ret, frame = cap.read()
    if not ret:
        break

    # Step 3: Flip frame for mirror effect
    frame = cv2.flip(frame, 1)

    # Step 4: Define region of interest (ROI) for hand
    roi = frame[100:400, 100:400]
    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    # Step 5: Convert ROI to HSV color space
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Step 6: Define skin color range and create mask
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Step 7: Apply morphological transformations
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=4)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)

    # Step 8: Find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=lambda x: cv2.contourArea(x))
        hull = cv2.convexHull(cnt)
        cv2.drawContours(roi, [hull], -1, (0, 255, 0), 2)

        # Step 9: Count defects (finger gaps)
        hull_indices = cv2.convexHull(cnt, returnPoints=False)
        defects = cv2.convexityDefects(cnt, hull_indices)

        if defects is not None:
            count_defects = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                cv2.circle(roi, far, 5, (0, 0, 255), -1)
                count_defects += 1

            # Step 10: Map gestures to actions
            if count_defects == 0:
                cv2.putText(frame, "Medicine: Paracetamol", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif count_defects == 1:
                cv2.putText(frame, "Medicine: Ibuprofen", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif count_defects == 2:
                cv2.putText(frame, "Medicine: Aspirin", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Step 11: Show frames
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    # Step 12: Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
