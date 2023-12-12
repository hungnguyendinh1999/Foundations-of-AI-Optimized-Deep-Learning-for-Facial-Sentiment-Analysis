import cv2
import torch
import os
from facenet_pytorch import MTCNN
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from Member import Member
from torchvision.transforms import transforms
from PIL import Image

# Initialize facenet face detector
face_detector = MTCNN(select_largest=False, device='cuda' if torch.cuda.is_available() else 'cpu')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

genes = torch.load(os.path.join("..", "res", "fer2013Saves", "population", "best_member_properties.pt"))["genes"]
emotion_model = Member(genes=genes).model
emotion_model.load_state_dict(
    torch.load(os.path.join("..", "res", "fer2013Saves", "population", "best_member_model.pt")))
emotion_model.to(device)
emotion_model.eval()

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

# Get webcam input
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect Faces
    boxes, _ = face_detector.detect(frame)

    if boxes is not None:
        for box in boxes:
            extracted_face = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            face_tensor = preprocess_face(extracted_face)

            # Classify Emotion
            with torch.no_grad():
                emotion_prediction = emotion_model(face_tensor).argmax(dim=1)
                emotion_label = emotion_labels[emotion_prediction.item()]

            # Visualization: Draw box and label
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
