import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model/modelv2.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define the ASL alphabet mapping
asl_alphabet = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # ['A', 'B', ..., 'Z']
asl_alphabet =asl_alphabet+['del','nothing','<space>']

# Preprocess the frame for the model
def preprocess_frame(frame, target_size=(128, 128)):
    frame_resized = cv2.resize(frame, target_size)
    frame_normalized = frame_resized / 255.0  # Normalize the image
    return np.expand_dims(frame_normalized, axis=0).astype(np.float32)  # Ensure float32

# Run the real-time ASL recognition from the webcam
cap = cv2.VideoCapture(0)  # Open webcam (0 is usually the default camera)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame and set input tensor
    input_data = preprocess_frame(frame)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output tensor and make a prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = asl_alphabet[np.argmax(output_data)]

    # Display the predicted class in red letters on the top-left of the frame
    cv2.putText(frame, f"Predicted Class: {predicted_class}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the frame with the prediction
    cv2.imshow('ASL Recognition', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()