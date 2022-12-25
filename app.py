import streamlit as st
import numpy as np
import cv2 as cv
from PIL import Image,ImageOps
import pickle
import pandas as pd
from sklearn.neighbors import NearestNeighbors

df = pd.read_csv('data.csv')
df['vector'] = df['vector'].apply(lambda x: np.asarray(x.replace('\n', '').replace('[','').replace(']','').split()).astype(float))

NNmodel = NearestNeighbors(n_neighbors=5,
                         metric='cosine',
                         algorithm='brute',
                         n_jobs=-1)
NNmodel.fit(np.stack(df['vector']))

sift = cv.SIFT_create()

with open("k_means.pkl", "rb") as f:
    k_means = pickle.load(f)

def makeHist(uploaded_file):
    res = np.zeros(1024)
    image = Image.open(uploaded_file).convert("RGB")

    img = ImageOps.exif_transpose(image)
    img = img.save("img.jpg")
    img = cv.imread("img.jpg")
    grayImg = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    points, descs = sift.detectAndCompute(grayImg,None)
    
    if len(points) == 0:
        return np.array([])
    
    kmeansPredict = k_means.predict(descs.astype(float))
    hist = np.histogram(kmeansPredict, bins=1024)[0]
    return hist / np.linalg.norm(hist)

filePath = st.file_uploader("")

if filePath is not None:
    hist = makeHist(filePath)
    for idImg in (NNmodel.kneighbors([hist])[1][0]):
        print("\n\n\n\n\n\n\n===========>  ", idImg, "\n\n\n\n\n\n\n")
        pathImg = df[df['id'] == idImg]['path']
        image = Image.open(pathImg.astype('string').values[0])
        st.image(image)
