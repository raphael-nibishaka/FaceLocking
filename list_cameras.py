"""
Find which camera indices are available (e.g. 0 = first, 1 = second).
Run: python list_cameras.py

Use the index you want with face_lock: python face_lock.py --camera 0
"""
import cv2

def main():
    print("Checking camera indices 0-9...")
    available = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                available.append((i, w, h))
    if not available:
        print("No cameras found. Check that a webcam is connected.")
        return
    print("\nAvailable cameras (index, width, height):")
    for idx, w, h in available:
        print(f"  Index {idx}: {w}x{h}  -> use: python face_lock.py --camera {idx}")
    print("\nTypical: 0 = first (often built-in), 1 = second (often USB webcam).")
    print("Pick the index that shows the camera you want.")

if __name__ == "__main__":
    main()
