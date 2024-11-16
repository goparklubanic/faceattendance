import base64
import os
import cv2
import face_recognition
import requests

# Set up paths
LIBRARY_FOLDER = "face_library/"
PAYROLL_SERVER_URL = os.getenv("PAYROLL_SERVER_URL")

def save_face_image(user_id, face_image_base64):
    """Save face image as a file in the library folder."""
    if not os.path.exists(LIBRARY_FOLDER):
        os.makedirs(LIBRARY_FOLDER)

    image_data = base64.b64decode(face_image_base64)
    image_path = os.path.join(LIBRARY_FOLDER, f"{user_id}.jpg")

    with open(image_path, "wb") as f:
        f.write(image_data)

    return image_path

def recognize_face(face_image_base64):
    """Recognize a face and return user ID if recognized."""
    image_data = base64.b64decode(face_image_base64)
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process known faces
    for file in os.listdir(LIBRARY_FOLDER):
        known_face = face_recognition.load_image_file(os.path.join(LIBRARY_FOLDER, file))
        known_face_encoding = face_recognition.face_encodings(known_face)[0]

        # Process unknown face from the frame
        unknown_face_encoding = face_recognition.face_encodings(frame)
        if unknown_face_encoding:
            matches = face_recognition.compare_faces([known_face_encoding], unknown_face_encoding[0])

            if True in matches:
                return os.path.splitext(file)[0]  # Return user ID (filename without extension)

    return None

def send_to_payroll(user_id, timestamp):
    """Send attendance record to payroll server."""
    data = {"user_id": user_id, "timestamp": timestamp}
    response = requests.post(PAYROLL_SERVER_URL, json=data)
    return response.ok
