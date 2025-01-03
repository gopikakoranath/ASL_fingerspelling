import cv2
import torch
import torchvision.transforms as transforms
from models import mobilenet_model  # Import your model definition

# Define the ASL alphabet mapping
asl_alphabet = [chr(i) for i in range(ord('A'), ord('Z') + 1)]  # ['A', 'B', ..., 'Z']
asl_alphabet =asl_alphabet+['del','nothing','<space>']

# Load the trained model
model = mobilenet_model.get_mobilenet_model(num_classes=29)  # Ensure the architecture matches the trained model
model.load_state_dict(torch.load("models/saved_model.pth", map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Define image preprocessing transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Open webcam for live feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

print("Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame
    input_frame = cv2.flip(frame, 1)  # Flip horizontally for a mirror-like view
    roi = input_frame[100:600, 500:1000]  # Crop the region of interest
    tensor = transform(roi).unsqueeze(0)  # Add batch dimension

    # Perform prediction
    with torch.no_grad():
        outputs = model(tensor)
        _, predicted_class = outputs.max(1)  # Get the index of the highest score
        predicted_label = asl_alphabet[predicted_class.item()]
        # predicted_label = predicted_class.item()

    # Display prediction on the frame
    cv2.putText(input_frame, f"Prediction: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.rectangle(input_frame, (500, 100), (1000, 600), (255, 0, 0), 2)  # Draw ROI rectangle
    cv2.imshow("ASL Gesture Recognition", input_frame)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()