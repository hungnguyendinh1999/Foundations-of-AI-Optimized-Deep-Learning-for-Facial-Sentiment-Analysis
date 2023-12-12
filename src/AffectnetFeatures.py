import cv2
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from sklearn.utils import shuffle
from tqdm import tqdm

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')

TARGET_SIZE = 160
FACENET_MODEL = InceptionResnetV1(pretrained='vggface2').eval().to(device)
LABEL_DICT = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "surprise": 3,
    "fear": 4,
    "disgust": 5,
    "anger": 6
}

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
PREPROCESS = transforms.Compose([transforms.Resize(TARGET_SIZE), transforms.ToTensor(), NORMALIZE])


def load_image_tensor(path):
    image = cv2.imread("../res/AffectnetData/" + path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image_tensor = PREPROCESS(image)
    return image_tensor


def extract_features(model, image_tensors, batch_size=64):
    features_list = []
    for i in tqdm(range(0, len(image_tensors), batch_size)):
        batch = image_tensors[i:i + batch_size].to(device)
        with torch.no_grad():
            batch_features = model(batch)
            features_list.append(batch_features.cpu())
    return torch.cat(features_list, dim=0)


def one_hot_encode(labels):
    one_hot = torch.zeros(labels.shape[0], len(LABEL_DICT))
    for i, label in enumerate(labels):
        one_hot[i][label] = 1
    return one_hot


df = pd.read_csv('../res/AffectnetData/labels.csv')
df = df[df['label'] != 'contempt']
df = df.dropna()
df = shuffle(df).reset_index(drop=True)

neutral_df = df[df['label'] == 'neutral']
happy_df = df[df['label'] == 'happy']
sad_df = df[df['label'] == 'sad']
surprise_df = df[df['label'] == 'surprise']
fear_df = df[df['label'] == 'fear']
disgust_df = df[df['label'] == 'disgust']
anger_df = df[df['label'] == 'anger']

neutral_images_tensor = torch.stack(neutral_df['pth'].apply(load_image_tensor).tolist())
neutral_features_tensor = extract_features(FACENET_MODEL, neutral_images_tensor)
neutral_labels_tensor = torch.tensor(neutral_df['label'].apply(lambda x: LABEL_DICT[x]).tolist())
neutral_one_hot_labels_tensor = one_hot_encode(neutral_labels_tensor)

torch.save(neutral_features_tensor, "../res/AffectnetData/neutral_features.pt")
torch.save(neutral_one_hot_labels_tensor, "../res/AffectnetData/neutral_labels.pt")

happy_images_tensor = torch.stack(happy_df['pth'].apply(load_image_tensor).tolist())
happy_features_tensor = extract_features(FACENET_MODEL, happy_images_tensor)
happy_labels_tensor = torch.tensor(happy_df['label'].apply(lambda x: LABEL_DICT[x]).tolist())
happy_one_hot_labels_tensor = one_hot_encode(happy_labels_tensor)

torch.save(happy_features_tensor, "../res/AffectnetData/happy_features.pt")
torch.save(happy_one_hot_labels_tensor, "../res/AffectnetData/happy_labels.pt")

sad_images_tensor = torch.stack(sad_df['pth'].apply(load_image_tensor).tolist())
sad_features_tensor = extract_features(FACENET_MODEL, sad_images_tensor)
sad_labels_tensor = torch.tensor(sad_df['label'].apply(lambda x: LABEL_DICT[x]).tolist())
sad_one_hot_labels_tensor = one_hot_encode(sad_labels_tensor)

torch.save(sad_features_tensor, "../res/AffectnetData/sad_features.pt")
torch.save(sad_one_hot_labels_tensor, "../res/AffectnetData/sad_labels.pt")

surprise_images_tensor = torch.stack(surprise_df['pth'].apply(load_image_tensor).tolist())
surprise_features_tensor = extract_features(FACENET_MODEL, surprise_images_tensor)
surprise_labels_tensor = torch.tensor(surprise_df['label'].apply(lambda x: LABEL_DICT[x]).tolist())
surprise_one_hot_labels_tensor = one_hot_encode(surprise_labels_tensor)

torch.save(surprise_features_tensor, "../res/AffectnetData/surprise_features.pt")
torch.save(surprise_one_hot_labels_tensor, "../res/AffectnetData/surprise_labels.pt")

fear_images_tensor = torch.stack(fear_df['pth'].apply(load_image_tensor).tolist())
fear_features_tensor = extract_features(FACENET_MODEL, fear_images_tensor)
fear_labels_tensor = torch.tensor(fear_df['label'].apply(lambda x: LABEL_DICT[x]).tolist())
fear_one_hot_labels_tensor = one_hot_encode(fear_labels_tensor)

torch.save(fear_features_tensor, "../res/AffectnetData/fear_features.pt")
torch.save(fear_one_hot_labels_tensor, "../res/AffectnetData/fear_labels.pt")

disgust_images_tensor = torch.stack(disgust_df['pth'].apply(load_image_tensor).tolist())
disgust_features_tensor = extract_features(FACENET_MODEL, disgust_images_tensor)
disgust_labels_tensor = torch.tensor(disgust_df['label'].apply(lambda x: LABEL_DICT[x]).tolist())
disgust_one_hot_labels_tensor = one_hot_encode(disgust_labels_tensor)

torch.save(disgust_features_tensor, "../res/AffectnetData/disgust_features.pt")
torch.save(disgust_one_hot_labels_tensor, "../res/AffectnetData/disgust_labels.pt")

anger_images_tensor = torch.stack(anger_df['pth'].apply(load_image_tensor).tolist())
anger_features_tensor = extract_features(FACENET_MODEL, anger_images_tensor)
anger_labels_tensor = torch.tensor(anger_df['label'].apply(lambda x: LABEL_DICT[x]).tolist())
anger_one_hot_labels_tensor = one_hot_encode(anger_labels_tensor)

torch.save(anger_features_tensor, "../res/AffectnetData/anger_features.pt")
torch.save(anger_one_hot_labels_tensor, "../res/AffectnetData/anger_labels.pt")
