import pandas as pd
import numpy as np
from glob import glob
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import dlib
from scipy.spatial import distance as dist
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns


def eye_aspect_ratio(eye):
    a = dist.euclidean(eye[1], eye[5])
    b = dist.euclidean(eye[2], eye[4])
    c = dist.euclidean(eye[0], eye[3])
    return (a+b)/(2*c) if c != 0 else 0

def mouth_opening_ratio(mouth):
  a = dist.euclidean(mouth[2], mouth[10])
  b = dist.euclidean(mouth[3], mouth[9])
  c = dist.euclidean(mouth[4], mouth[8])
  d = dist.euclidean(mouth[0], mouth[6])
  return (a+b+c)/(3*d) if d != 0 else 0

def nose_length_ratio(nose_top,nose_tip,chin_point):
  a = dist.euclidean(nose_top, nose_tip)
  b = dist.euclidean(nose_tip, chin_point)
  return a/b if b != 0 else 0

class DrowsinessDataset(Dataset):
  def __init__(self,image_paths,labels,detector,predictor):
    self.image_paths = image_paths
    self.labels = labels
    self.detector = detector
    self.predictor = predictor

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self,idx):
    image_path = self.image_paths[idx]
    image = cv2.imread(self.image_paths[idx])
    image_resized = cv2.resize(image,(128,128))
    image_tensor = torch.tensor(image_resized).permute(2,0,1).float()/255.0

    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    faces = self.detector(gray)
    if len(faces) == 0:
        features_tensor = torch.zeros(4).float()
        label = torch.tensor([0.0])
        return image_tensor, features_tensor, label
    landmarks_xy = []
    landmarks = self.predictor(gray,faces[0])
    landmarks_xy = [(landmarks.part(i).x,landmarks.part(i).y) for i in range(68)]
    left_eye = landmarks_xy[36:42]
    right_eye = landmarks_xy[42:48]
    mouth = landmarks_xy[48:68]
    nose_top = landmarks_xy[27]
    nose_tip = landmarks_xy[30]
    chin_point = landmarks_xy[8]

    ear_left = eye_aspect_ratio(left_eye)
    ear_right = eye_aspect_ratio(right_eye)
    mor = mouth_opening_ratio(mouth)
    nlr = nose_length_ratio(nose_top,nose_tip,chin_point)

    features_tensor = torch.tensor([ear_left, ear_right, mor, nlr]).float()
    label = torch.tensor([1.0] if self.labels[idx] == 'Drowsy' else [0.0])
    return image_tensor,features_tensor,label

class DowsinessDetectorCNN(nn.Module):
  def __init__(self):
    super(DowsinessDetectorCNN,self).__init__()

    self.model = nn.Sequential(

        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Flatten()
    )

    self.fc_image = nn.Linear(128*16*16,512)
    self.fc_features = nn.Linear(4,16)
    self.fc1 = nn.Linear(512+16,256)
    self.fc2 = nn.Linear(256,1)

  def forward(self,image,features):
    image_features = self.model(image)
    image_features = torch.relu(self.fc_image(image_features))
    additional_features = torch.relu(self.fc_features(features))
    combined_features = torch.cat((image_features,additional_features),dim=1)
    combined_features = torch.relu(self.fc1(combined_features))
    output = torch.sigmoid(self.fc2(combined_features))
    return output

def TrainCNN(model,train_loader,optimizer,loss_fn,device):
  num_epochs = 50
  train_loss_list = []

  for epoch in range(num_epochs):
    train_loss = 0
    model.train()

    for i, (images,features,labels) in enumerate(train_loader):
      images =images.to(device)
      features = features.to(device)
      labels = labels.to(device)

      outputs = model(images,features)
      loss = loss_fn(outputs,labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      train_loss += loss.item()

    train_loss_list.append(train_loss/len(train_loader))

  plt.plot(range(1,num_epochs+1), train_loss_list)
  plt.xlabel("Number of epochs")
  plt.ylabel("Training loss")

def EvaluateCNN(model,test_loader,device):
  test_val = 0
  model.eval()
  predictions_list = []
  labels_list = []
  with torch.no_grad():
    for i,(images,features,labels) in enumerate(test_loader):
      images  = images.to(device)
      features = features.to(device)
      y_test = labels.to(device)

      outputs = model(images,features)
      preds = (outputs > 0.5).float()
      test_val += (preds == y_test).sum().item()
      predictions_list.extend(preds.cpu().numpy())
      labels_list.extend(labels.cpu().numpy())

  test_accuracy = test_val/len(test_loader.dataset)
  return test_accuracy,np.array(predictions_list),np.array(labels_list)

non_drowsy = glob('/content/drive/My Drive/Non Drowsy/*')
drowsy = glob('/content/drive/My Drive/Drowsy/*')

data = {'image': drowsy+non_drowsy,'label': ['Drowsy']*len(drowsy)+['Non Drowsy']*len(non_drowsy)}

image_df = pd.DataFrame(data)

X_train,X_test,y_train,y_test = train_test_split(image_df['image'],image_df['label'],test_size=0.2,random_state=42,shuffle = True)

path = '/content/shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(path)

training_data = DrowsinessDataset(X_train.values,y_train.values,detector,predictor)
testing_data = DrowsinessDataset(X_test.values,y_test.values,detector,predictor)
training_loader = DataLoader(training_data,batch_size=32,shuffle=True)
testing_loader = DataLoader(testing_data,batch_size=32,shuffle=False)

device ='cuda' if torch.cuda.is_available() else 'cpu'
model = DowsinessDetectorCNN().to(device)
learning_rate = 0.001
weight_decay = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)

TrainCNN(model,training_loader,optimizer,criterion,device)

accuracy, predictions, actual = EvaluateCNN(model,testing_loader,device)
cm = confusion_matrix(actual, predictions)
print(f'Model Accuracy: {accuracy*100}%')
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Drowsy', 'Drowsy'], yticklabels=['Not Drowsy', 'Drowsy'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()