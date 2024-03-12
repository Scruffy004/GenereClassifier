import librosa
import pandas as pd
import numpy as np
import ffmpeg
from yt_dlp import YoutubeDL
from pydub import AudioSegment
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
import keras
from keras import models, layers

df = pd.read_csv('data-3s.csv')

seed = 12

np.random.seed(seed)

shuffle = df.sample(frac=1, random_state=seed).reset_index(drop=True)

shuffle = shuffle.drop(['filename'], axis=1)

X = shuffle.iloc[:, :-1]
genres = shuffle.iloc[:, -1]
y = LabelEncoder().fit_transform(genres)

X_train, df_test_valid_X, y_train, df_test_valid_y = train_test_split(X, y, train_size=0.7, random_state=seed,
                                                                      stratify=y)
X_dev, X_test, y_dev, y_test = train_test_split(df_test_valid_X, df_test_valid_y, train_size=0.66, random_state=seed,
                                                stratify=df_test_valid_y)

scaler = StandardScaler()

X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_dev = pd.DataFrame(scaler.transform(X_dev), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)




model = keras.models.Sequential([keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
                                 keras.layers.Dense(128, activation='relu'),
                                 keras.layers.Dense(64, activation='relu'),
                                 keras.layers.Dense(10, activation='softmax')])
print(model.summary())

model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.SparseCategoricalCrossentropy(), metrics='accuracy')

history = model.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=70, batch_size=128)

test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=128)
print("The test Loss is :", test_loss)
print("\nThe Best test Accuracy is :", test_acc * 100)

ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': 'output.%(ext)s',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
    }],
}

with YoutubeDL(ydl_opts) as ydl:
    url = input("Enter URL: ")
    ydl.download([url])
    stream = ffmpeg.input('output.m4a')
    stream = ffmpeg.output(stream, 'output.wav')

t1 = 60000
t2 = 90000

waveFile = AudioSegment.from_file('output.wav')
waveFile = waveFile[t1:t2]
waveFile.export('output30s.wav', format="wav")

y, sr = librosa.load('output30s.wav', mono=True, duration=30)
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
rmse = librosa.feature.rms(y=y)
spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
zcr = librosa.feature.zero_crossing_rate(y)
mfcc = librosa.feature.mfcc(y=y, sr=sr)
feature = np.array([np.mean(chroma_stft), np.mean(rmse), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)])
for e in mfcc:
        feature = np.append(feature, [np.mean(e)])
y = model.predict(scaler.transform([feature]))
print(y)
i = np.argmax(y)
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
print(genres[i])

