from flask import Flask, render_template, Response
import cv2
from detection import detect_faces

app = Flask(__name__)

# Global variable for video capture
video_capture = cv2.VideoCapture(0)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

def generate_frames():
    """Generate video frames with face detection."""
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        else:
            # Process the frame for face detection
            frame = detect_faces(frame)

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
