import pandas as pd
import cv2
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

from tqdm import tqdm
from sklearn.utils import shuffle
from PIL import Image
from facenet_pytorch import InceptionResnetV1

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

images_tensor = torch.stack(df['pth'].apply(load_image_tensor).tolist())
features_tensor = extract_features(FACENET_MODEL, images_tensor)
labels_tensor = torch.tensor(df['label'].apply(lambda x: LABEL_DICT[x]).tolist())
one_hot_labels_tensor = one_hot_encode(labels_tensor)
print(one_hot_labels_tensor.shape)

torch.save(features_tensor, '../res/AffectnetData/features.pt')
torch.save(one_hot_labels_tensor, '../res/AffectnetData/labels.pt')
