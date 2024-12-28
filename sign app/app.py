from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as mp
import numpy as np
import pickle

app = Flask(__name__)
socketio = SocketIO(app)

# Load the trained model
try:
    with open('model.p', 'rb') as model_file:
        model_dict = pickle.load(model_file)
        model = model_dict['model']
except FileNotFoundError:
    raise FileNotFoundError("The file 'model.p' was not found. Please ensure it exists in the project directory.")
except KeyError:
    raise KeyError("The loaded model file does not contain the expected key 'model'.")

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define label mapping
labels_dict = {0: 'A', 1: 'B', 2: 'L'}

@app.route('/')
def index():
    """Renders the main page."""
    return render_template('index.html')

def process_frame():
    """Processes video frames from the webcam for gesture recognition."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Unable to read frame from the webcam.")
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # Process detected hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x_ = [lm.x for lm in hand_landmarks.landmark]
                    y_ = [lm.y for lm in hand_landmarks.landmark]

                    data_aux = []
                    for lm in hand_landmarks.landmark:
                        data_aux.append(lm.x - min(x_))
                        data_aux.append(lm.y - min(y_))

                    # Predict the gesture using the loaded model
                    try:
                        prediction = model.predict([np.asarray(data_aux)])
                        predicted_character = labels_dict[int(prediction[0])]
                    except Exception as e:
                        print(f"Error during prediction: {e}")
                        continue

                    # Emit the prediction to the frontend
                    socketio.emit('prediction', {'character': predicted_character})

            # Display the video frame
            cv2.imshow('Gesture Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

@socketio.on('connect')
def handle_connect():
    """Handles a new client connection and starts the background task."""
    print("Client connected.")
    socketio.start_background_task(process_frame)

if __name__ == '__main__':
    try:
        socketio.run(app, debug=True)
    except Exception as e:
        print(f"An error occurred: {e}")
