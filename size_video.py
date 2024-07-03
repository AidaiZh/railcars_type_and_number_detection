
import cv2

# Initialize the video capture
cap = cv2.VideoCapture("converted_video_06_27/18.55.00-18.58.08[A][0@0][0].mp4")
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()
mask=cv2.imread("vc_mask.png")
# Get the width and height of the video frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Frame width: {frame_width}")
print(f"Frame height: {frame_height}")
print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")

# Release the video capture
cap.release()
