import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Failed to open camera index 0')
    exit(1)

print('Camera opened')
ret, frame = cap.read()
if not ret or frame is None:
    print('Failed to read first frame')
    cap.release()
    exit(1)

print('First frame size:', frame.shape[1], 'x', frame.shape[0])
cv2.imwrite('test_debug.jpg', frame)
print('Wrote test_debug.jpg')
cv2.imshow('Raw', frame)
print("Press any key in the window to exit")
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
