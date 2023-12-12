import os

import numpy as np
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
    0: 'anger',
    1: 'disgust',
    2: 'fear',
    3: 'happiness',
    4: 'sadness',
    5: 'surprise',
    6: 'neutral'
}

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
PREPROCESS = transforms.Compose([transforms.Resize(TARGET_SIZE), transforms.ToTensor(), NORMALIZE])


def load_image_tensor(array):
    array = array.reshape(48, 48)
    image = Image.fromarray(array)
    image = image.convert('RGB')
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


df = pd.read_csv(os.path.join('..', 'res', 'fer2013', 'fer2013.csv'))
df = df.dropna()
df = shuffle(df).reset_index(drop=True)

df['pixels'] = df['pixels'].apply(lambda x: np.array(x.split(' ')).astype(np.float32))

neutral_df = df[df['emotion'] == 6]

surprise_df = df[df['emotion'] == 5]
sad_df = df[df['emotion'] == 4]
happy_df = df[df['emotion'] == 3]
fear_df = df[df['emotion'] == 2]
disgust_df = df[df['emotion'] == 1]
anger_df = df[df['emotion'] == 0]


neutral_images_tensor = torch.stack(neutral_df['pixels'].apply(load_image_tensor).tolist())
neutral_features_tensor = extract_features(FACENET_MODEL, neutral_images_tensor)
neutral_labels_tensor = torch.tensor(neutral_df['emotion'].tolist())
neutral_one_hot_labels_tensor = one_hot_encode(neutral_labels_tensor)

torch.save(neutral_features_tensor, os.path.join('..', 'res', 'fer2013', 'neutral_features.pt'))
torch.save(neutral_one_hot_labels_tensor, os.path.join('..', 'res', 'fer2013', 'neutral_labels.pt'))

happy_images_tensor = torch.stack(happy_df['pixels'].apply(load_image_tensor).tolist())
happy_features_tensor = extract_features(FACENET_MODEL, happy_images_tensor)
happy_labels_tensor = torch.tensor(happy_df['emotion'].tolist())
happy_one_hot_labels_tensor = one_hot_encode(happy_labels_tensor)

torch.save(happy_features_tensor, os.path.join('..', 'res', 'fer2013', 'happy_features.pt'))
torch.save(happy_one_hot_labels_tensor, os.path.join('..', 'res', 'fer2013', 'happy_labels.pt'))

sad_images_tensor = torch.stack(sad_df['pixels'].apply(load_image_tensor).tolist())
sad_features_tensor = extract_features(FACENET_MODEL, sad_images_tensor)
sad_labels_tensor = torch.tensor(sad_df['emotion'].tolist())
sad_one_hot_labels_tensor = one_hot_encode(sad_labels_tensor)

torch.save(sad_features_tensor, os.path.join('..', 'res', 'fer2013', 'sad_features.pt'))
torch.save(sad_one_hot_labels_tensor, os.path.join('..', 'res', 'fer2013', 'sad_labels.pt'))

surprise_images_tensor = torch.stack(surprise_df['pixels'].apply(load_image_tensor).tolist())
surprise_features_tensor = extract_features(FACENET_MODEL, surprise_images_tensor)
surprise_labels_tensor = torch.tensor(surprise_df['emotion'].tolist())
surprise_one_hot_labels_tensor = one_hot_encode(surprise_labels_tensor)

torch.save(surprise_features_tensor, os.path.join('..', 'res', 'fer2013', 'surprise_features.pt'))
torch.save(surprise_one_hot_labels_tensor, os.path.join('..', 'res', 'fer2013', 'surprise_labels.pt'))

fear_images_tensor = torch.stack(fear_df['pixels'].apply(load_image_tensor).tolist())
fear_features_tensor = extract_features(FACENET_MODEL, fear_images_tensor)
fear_labels_tensor = torch.tensor(fear_df['emotion'].tolist())
fear_one_hot_labels_tensor = one_hot_encode(fear_labels_tensor)

torch.save(fear_features_tensor, os.path.join('..', 'res', 'fer2013', 'fear_features.pt'))
torch.save(fear_one_hot_labels_tensor, os.path.join('..', 'res', 'fer2013', 'fear_labels.pt'))

disgust_images_tensor = torch.stack(disgust_df['pixels'].apply(load_image_tensor).tolist())
disgust_features_tensor = extract_features(FACENET_MODEL, disgust_images_tensor)
disgust_labels_tensor = torch.tensor(disgust_df['emotion'].tolist())
disgust_one_hot_labels_tensor = one_hot_encode(disgust_labels_tensor)

torch.save(disgust_features_tensor, os.path.join('..', 'res', 'fer2013', 'disgust_features.pt'))
torch.save(disgust_one_hot_labels_tensor, os.path.join('..', 'res', 'fer2013', 'disgust_labels.pt'))

anger_images_tensor = torch.stack(anger_df['pixels'].apply(load_image_tensor).tolist())
anger_features_tensor = extract_features(FACENET_MODEL, anger_images_tensor)
anger_labels_tensor = torch.tensor(anger_df['emotion'].tolist())
anger_one_hot_labels_tensor = one_hot_encode(anger_labels_tensor)

torch.save(anger_features_tensor, os.path.join('..', 'res', 'fer2013', 'anger_features.pt'))
torch.save(anger_one_hot_labels_tensor, os.path.join('..', 'res', 'fer2013', 'anger_labels.pt'))
