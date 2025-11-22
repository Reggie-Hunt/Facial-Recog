import cv2

# Use the full path to your downloaded XML file
# Replace with your path if different
cascade_path = "haarcascade_frontalface_alt.xml"
face_classifier = cv2.CascadeClassifier(cascade_path)

# Check if the cascade loaded correctly
if face_classifier.empty():
    raise IOError("Failed to load Haar cascade XML file. Check the path!")


# Convert image to grayscale for face detection
def greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in an image
def classify(image):
    # detectMultiScale parameters:
    # scaleFactor: how much the image size is reduced at each scale
    # minNeighbors: how many neighbors each candidate rectangle should have to retain it
    # minSize: minimum possible object size
    return face_classifier.detectMultiScale(
        greyscale(image),
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

# Draw rectangles around detected faces
def bound(image):
    faces = classify(image)
    if faces is not None:
        for (x, y, w, h) in faces:
            # Draw rectangle: (image, top-left, bottom-right, color BGR, thickness)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 4)

# Prepare the image for display
def show(image):
    bound(image)  # draw rectangles on detected faces
    return image   # return BGR image for cv2.imshow

cv2.namedWindow("WebCam")          # create a named window
window = cv2.VideoCapture(0)           # open default webcam (0)

if window.isOpened():
    fo, frame = window.read()        # read the first frame
else:
    fo = False


#Main webcam loop


while open:
    if frame is not None:
        cv2.imshow("WebCam", show(frame))  # show processed frame with rectangles

    open, frame = window.read()        # read next frame
    key = cv2.waitKey(10)          # wait 20ms for a key press
    if key == 27:                  # exit loop if ESC is pressed
        break


cv2.destroyWindow("WebCam")       # close the window
window.release()                       # release the webcam
 