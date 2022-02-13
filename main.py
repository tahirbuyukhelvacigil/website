import streamlit as st
import numpy as np
import librosa
import librosa.display
from pydub import AudioSegment
import pickle
from  sklearn.preprocessing import StandardScaler,MinMaxScaler
st.title("Akciğer Seslerindeki Anomali Tespit Uygulaması")
st.header("Bu Web Uygulaması İle Akciğer Sesleri Sınıflandırılabilmektedir.")
st.sidebar.title("Akciğer Seslerindeki Anomali Tespit Uygulaması")
st.sidebar.subheader("Daha Doğru Sonuçlar Elde Etmek İçin Yükleyeceğiniz Ses Dosyasının Aşağıdaki Ses Dosyası Gibi Olmasına Dikkat Ediniz.")
audio_file = open(r'sagliki-derin-nefes (13).wav', 'rb')
audio_bytes = audio_file.read()
st.sidebar.audio(audio_bytes, format='audio/wav')
st.sidebar.subheader("Sınıflandırma İşleminin Gerçekleşebilmesi İçin Aşağıdaki Bölüme Wav  Formatındaki Ses Dosyasını Yükleyiniz.Ses Dosyasının 3-7 Saniye Arasında Olması Önerilmektedir.")
file = st.sidebar.file_uploader("",type=["wav"])
model = pickle.load(open("akciger_ses4.sav", 'rb'))
st.sidebar.title('Sağlıklı Günler Dileriz!')

t1 = 2
t2 = 7
def extract(file, t1, t2):
 wav = AudioSegment.from_wav(file)
 wav = wav[1000 * t1:1000 * t2]
 wav.export('islenmis.wav', format='wav')


def siniflama(file):
 file_path = file
 X, sample_rate = librosa.load(file_path)
 sample_rate = np.array(sample_rate)

 n_fft = int(sample_rate * 0.025)
 hop_length = int(sample_rate * 0.01)
 mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40), axis=0)
 mfcc = librosa.feature.mfcc(y=X, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mfcc=40)
 mean_mfcc = np.mean(mfcc.T, axis=0)
 std_mfcc = np.std(mfcc.T, axis=0)
 dat = np.vstack((mean_mfcc, std_mfcc))
 data = [dat]
 data = np.asarray(data)
 data = data.reshape([data.shape[1] * data.shape[2], 1])
 scaler = StandardScaler()
 scaler.fit(data)
 data = scaler.transform(data)
 st.title("Sınıflandırma Sonucu")
 data = data.reshape(1, 80)
 output = model.predict(data)
 print(output)
 if output==[0.]:
  st.write("Gayet Sağlıklı Görünüyorsunuz :)")
  data = data.reshape(80, 1)
  st.title('Ses Dosyasının Vektörleri')
  data
 else:
  st.write("Hastalıklı Bir Ses :)")
  st.title('Ses Dosyasının Vektörleri')
  data = data.reshape(80, 1)
  data



if file is None:
 st.subheader('Lütfen Sınıflandırılacak Dosyayı Yükleyiniz.')
else:

 if st.button('Sınıflandır'):
    extract(file, t1, t2)
    audio_file = open('islenmis.wav', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')
    siniflama('islenmis.wav')














