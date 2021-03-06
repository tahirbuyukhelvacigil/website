import streamlit as st
import numpy as np
import librosa
import librosa.display
from pydub import AudioSegment
import ffmpeg
import pickle
from  sklearn.preprocessing import StandardScaler,MinMaxScaler
st.title("Akciğer Seslerindeki Anomali Tespit Uygulaması")
st.write(" Geliştirme Aşamasında Olan  Mobil Uygulamamıza  Ulaşmak İçin Tıklayınız https://drive.google.com/drive/u/0/folders/14-tre_u1dHj4pTj1v7NfZRkilSwJpE83")
st.write(" ")
st.write(" ")
st.markdown("Daha Doğru Sonuçlar :white_check_mark: Elde Etmek İçin Yükleyeceğiniz Ses Dosyasının :sound: Aşağıdaki Ses Dosyası Gibi Olmasına Dikkat Ediniz.")
audio_file = open(r'sagliki-derin-nefes (13).wav', 'rb')
audio_bytes = audio_file.read()
st.audio(audio_bytes, format='audio/wav')
st.markdown('Örnek Ses Kaydı **Coswara** Verisetinden Alınmıştır.Detaylı Bilgi için: https://github.com/iiscleap/Coswara-Data')
st.write(" ")
st.write(" ")
st.markdown("Sınıflandırma İşleminin Gerçekleşebilmesi İçin Aşağıdaki Bölüme Mp3  Formatındaki Ses Dosyasını Yükleyiniz. Ses Dosyasının **3-7 Saniye** Arasında Olması Önerilmektedir.")
file = st.file_uploader("",type=["mp3"])
model = pickle.load(open(r"akciger_ses4.sav", 'rb'))

def convert_mp3_to_wav(music_file):
 sound = AudioSegment.from_mp3(music_file)
 sound.export("islenmis.wav", format="wav")

t1 = 2
t2 = 7
def extract(file, t1, t2):
 wav = AudioSegment.from_wav(file)
 wav = wav[1000 * t1:1000 * t2]
 wav.export('islenmis2.wav', format='wav')

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
 st.title("Sınıflandırma Sonucu :white_check_mark:")
 data = data.reshape(1, 80)
 output = model.predict(data)
 print(output)
 if output==[0.]:
  st.write("Gayet Sağlıklı Görünüyorsunuz :smile: ")
  st.title('Sağlıklı Günler Dileriz! :scream: ')
 else:
  st.write("Hastalıklı Bir Ses :hospital:")
  st.title('Sağlıklı Günler Dileriz!:scream:')

if file is None:
 st.subheader('Lütfen Sınıflandırılacak Dosyayı Yükleyiniz.')
else:

 if st.button('Sınıflandır'):
    convert_mp3_to_wav(file)
    extract('islenmis.wav', t1, t2)
    audio_file = open('islenmis2.wav', 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')
    siniflama('islenmis2.wav')














