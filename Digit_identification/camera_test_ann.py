import cv2
import numpy as np
from tensorflow import keras

# Load trained MNIST model
model = keras.models.load_model("mnist_NN_model.keras", compile=False)

# Open webcam
cap = cv2.VideoCapture(0)
print("Press R to predict again | Q to quit")

predict_now = True
digit = None
confidence = 0.0

CONF_THRESHOLD = 0.95   # 95%



while True:
    ret, frame = cap.read() #frame(NumPy array) → image
    if not ret:
        break

    # Define Region of Interest (ROI)
    x1, y1, x2, y2 = 200, 100, 400, 300
    roi = frame[y1:y2, x1:x2]

    # Draw rectangle on main frame
    #cv2.rectangle(image, pt1(top-left), pt2(bottom-right), color, thickness)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


    if predict_now:

        # Convert ROI to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Blur to remove noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Threshold (BLACK pen on WHITE paper)
        #cv2.threshold(), Converts a grayscale image into a binary image (only black and white)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                #cv2.threshold(src, thresh, maxval, type) #cv2.THRESH_BINARY_INV => Digit → white,Background → black  # cv2.THRESH_OTSU  


        # Optional: make digits thicker for better recognition
        kernel = np.ones((2,2), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        # Resize to 28x28 (MNIST size)
        img = cv2.resize(thresh, (28, 28))

        # Normalize (0–1)
        img = img.astype("float32") / 255.0

        # MODEL EXPECTS (1, 28, 28)
        img = img.reshape(1, 28, 28)

        # Predict
        prediction = model.predict(img, verbose=0)
        temp_digit = np.argmax(prediction)
        temp_confidence = prediction[0][temp_digit]


        # Accept prediction ONLY if confidence >= 95%
        if temp_confidence >= CONF_THRESHOLD:
            digit = temp_digit
            confidence = temp_confidence
            predict_now = False   # PAUSE


    # Show prediction on the frame
    if digit is not None:
        cv2.putText(frame,
                    f"Digit: {digit}  Conf: {confidence:.2f}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2)
    else:
        cv2.putText(frame,
                    "Processing... Show digit clearly",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2)
    
    cv2.putText(frame,
                "Press R = Next | Q = Quit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2)

    # Show video
    cv2.imshow("Webcam Digit Recognition", frame)
   


    key = cv2.waitKey(1) & 0xFF
    # Exit when 'q' is pressed
    if key == ord('r') or key == ord('R'):
        predict_now = True
        digit = None
        confidence = 0.0

    elif key == ord('q') or key == ord('Q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
