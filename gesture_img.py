import cv2
import os

DATASET_PATH = "ASL_reference"
print("Press A-Z else ESC to quit")
cv2.namedWindow("ASL Gesture") # create a blank window
while True:
    key = cv2.waitKey(0)     # wait for key press
    # ESC to exit
    if key == 27:
        break
    letter = chr(key).upper() # convert key to character
    print(letter)
    if letter.isalpha():
        img_path = os.path.join(DATASET_PATH, f"{letter}.jpg")
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            cv2.imshow("ASL Gesture", img)
        else:
            print(f"{letter}.jpg not found")
cv2.destroyAllWindows()