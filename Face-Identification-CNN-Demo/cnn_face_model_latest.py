import cv2
import numpy as np
import tensorflow as tf
import json

# Load trained model
model = tf.keras.models.load_model("cnn_face_model_latest.keras", compile=False)

# Load label map
with open("cnn_face_model_latest_labels.json", "r") as f:
    label_map = json.load(f)
label_map = {int(k): v for k, v in label_map.items()}

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# Open webcam
cap = cv2.VideoCapture(0)
IMG_SIZE = 224  # match training input

if not cap.isOpened():
    print("Camera not opened")
    exit()

print("Camera opened. Press 'Q' to quit anytime.")

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Crop and preprocess face
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        # Prediction
        preds = model.predict(face, verbose=0)
        class_id = np.argmax(preds)
        confidence = np.max(preds)
        name = label_map.get(class_id, "Unknown")

        # Set box color
        color = (0, 255, 0) if confidence > 0.6 else (0, 0, 255)

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame,
            f"{name} ({confidence*100:.1f}%)",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    # Display the frame
    cv2.imshow("Face Recognition - Custom CNN", frame)

    # Quit if 'Q' pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
