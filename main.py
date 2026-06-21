import cv2
import numpy as np
import time


def try_open_camera(index=0):
    # Try a few common Windows backends and plain open as fallback
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW, None]
    for b in backends:
        try:
            cap = cv2.VideoCapture(index, b) if b is not None else cv2.VideoCapture(index)
        except Exception:
            cap = cv2.VideoCapture(index)
        if cap is None:
            continue
        if cap.isOpened():
            print(f"Opened camera index={index} backend={b}")
            return cap
        cap.release()
    return None


cap = try_open_camera(0)
if not cap:
    print("Failed to open camera index 0; trying indices 1..3")
    for i in range(1, 4):
        cap = try_open_camera(i)
        if cap:
            break

if not cap:
    print("Could not open any camera. Check permissions and close other apps using the camera.")
    exit(1)

print("Camera opened. Press 'q' to quit.")

first_info = True
fail_count = 0

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        fail_count += 1
        if fail_count > 50:
            print("Repeated frame read failures. Exiting.")
            break
        time.sleep(0.05)
        continue
    fail_count = 0

    if first_info:
        h, w = frame.shape[:2]
        print(f"Frame size: {w}x{h}")
        # Save first frame for offline inspection
        try:
            cv2.imwrite("debug_frame.jpg", frame)
            print("Wrote debug_frame.jpg")
        except Exception as e:
            print("Failed to write debug frame:", e)
        first_info = False

    # Show raw frame unmodified for diagnosis
    cv2.imshow("Raw", frame)

    frame = cv2.flip(frame, 1)

    # Safe ROI handling: clamp to frame size
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = 100, 100, 400, 400
    x2 = min(x2, w)
    y2 = min(y2, h)
    if x1 >= x2 or y1 >= y2:
        roi = frame.copy()
    else:
        roi = frame[y1:y2, x1:x2]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=4)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=lambda x: cv2.contourArea(x))
        if cv2.contourArea(cnt) > 1000:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(roi, [hull], -1, (0, 255, 0), 2)
            hull_indices = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt, hull_indices)

            if defects is not None:
                count_defects = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    far = tuple(cnt[f][0])
                    cv2.circle(roi, far, 5, (0, 0, 255), -1)
                    count_defects += 1

                if count_defects == 0:
                    cv2.putText(frame, "Medicine: Paracetamol", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif count_defects == 1:
                    cv2.putText(frame, "Medicine: Ibuprofen", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif count_defects == 2:
                    cv2.putText(frame, "Medicine: Aspirin", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()