import pandas as pd # type: ignore
import numpy as np # type: ignore
from glob import glob
import cv2 # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.model_selection import train_test_split # type: ignore

non_drowsy = glob('/Users/jiyamaheshwari/Downloads/Driver Drowsiness Dataset (DDD)/Non Drowsy/*')
drowsy= glob('/Users/jiyamaheshwari/Downloads/Driver Drowsiness Dataset (DDD)/Drowsy/*')

data = {'image': drowsy+non_drowsy,'label': ['Drowsy']*len(drowsy)+['Non Drowsy']*len(non_drowsy)}

image_df = pd.DataFrame(data)
image_df = image_df.sample(frac=1).reset_index(drop=True)

X_train,X_test,y_train,y_test = train_test_split(image_df['image'],image_df['label'],test_size=0.2,random_state=42)

print(image_df.head())

