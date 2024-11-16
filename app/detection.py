import cv2

# Load pre-trained face detector (Haar Cascades)
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

def detect_faces(frame):
    """
    Detect faces in a given frame.
    Args:
        frame: The frame from the webcam/video feed.
    Returns:
        frame: The annotated frame with rectangles around detected faces.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw rectangle around face

    return frame
