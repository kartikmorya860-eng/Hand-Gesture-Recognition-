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
    h, w = frame.shape[:2]

    # Use the full frame for detection, but draw a guidance box to help placement.
    x1, y1, x2, y2 = 50, 50, w - 50, h - 50
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    roi = frame.copy()

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    medicine_text = "Show hand inside green box"
    if contours:
        cnt = max(contours, key=lambda x: cv2.contourArea(x))
        area = cv2.contourArea(cnt)
        if area > 3000:
            hull = cv2.convexHull(cnt)
            cv2.drawContours(roi, [hull], -1, (0, 255, 0), 2)
            hull_indices = cv2.convexHull(cnt, returnPoints=False)
            defects = cv2.convexityDefects(cnt, hull_indices)

            count_defects = 0
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    if d > 5000:
                        far = tuple(cnt[f][0])
                        cv2.circle(roi, far, 5, (0, 0, 255), -1)
                        count_defects += 1

            if count_defects == 0:
                medicine_text = "Medicine: Paracetamol"
            elif count_defects == 1:
                medicine_text = "Medicine: Ibuprofen"
            elif count_defects == 2:
                medicine_text = "Medicine: Aspirin"
            else:
                medicine_text = "Medicine: Consult doctor"
        else:
            medicine_text = "Move hand closer to camera"

    cv2.putText(frame, medicine_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, "Press q to quit", (50, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()