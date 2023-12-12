import cv2
import torch
import os
from facenet_pytorch import MTCNN
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from Member import Member
from torchvision.transforms import transforms
from PIL import Image

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')

# Initialize facenet face detector
face_detector = MTCNN(device)

# Load emotion model
genes = torch.load(os.path.join("..", "res", "AffectnetSaves", "population", "best_member_properties.pt"),
                   map_location=device)["genes"]
emotion_model = Member(genes=genes).model
emotion_model.load_state_dict(
    torch.load(os.path.join("..", "res", "AffectnetSaves", "population", "best_member_model.pt"), map_location=device))
emotion_model.to(device)
emotion_model.eval()

# Load facenet model for feature extraction
facenet_model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']

def preprocess_face(face):
    preprocess = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        NORMALIZE
    ])

    # Convert to RGB and apply preprocessing
    face_rgb = Image.fromarray(face).convert('RGB')
    face_tensor = preprocess(face_rgb).to(device)

    # Extract features using InceptionResnetV1
    with torch.no_grad():
        face_features = facenet_model(face_tensor.unsqueeze(0))

    return face_features


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


# Get webcam input
cap = cv2.VideoCapture(0)

cv2.namedWindow('Emotion Detection', cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect Faces
    boxes, _ = face_detector.detect(frame)

    if boxes is not None:
        for box in boxes:
            # Extract coordinates
            x1, y1, x2, y2 = box

            # Clamp the box coordinates to be within the frame
            x1 = clamp(x1, 0, frame.shape[1])
            y1 = clamp(y1, 0, frame.shape[0])
            x2 = clamp(x2, 0, frame.shape[1])
            y2 = clamp(y2, 0, frame.shape[0])

            if (x2 - x1 <= 0) or (y2 - y1 <= 0):
                continue

            height, width = frame.shape[:2]
            cv2.resizeWindow('Emotion Detection', width, height)

            extracted_face = frame[int(y1):int(y2), int(x1):int(x2)]
            if extracted_face.size == 0:
                continue
            
            face_tensor = preprocess_face(extracted_face)

            # Classify Emotion
            with torch.no_grad():
                emotion_prediction = emotion_model(face_tensor).argmax(dim=1)
                emotion_label = emotion_labels[emotion_prediction.item()]

            # Draw box and label
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 1)
            cv2.putText(frame, emotion_label, (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 255), 1)

    # Display the resulting frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()
