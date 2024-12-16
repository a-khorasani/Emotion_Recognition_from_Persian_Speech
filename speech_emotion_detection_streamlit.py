import streamlit as st
import torch
import torchaudio
import librosa
from torchaudio import pipelines
import torch.nn as nn
import sounddevice as sd
import numpy as np
import warnings
import torchvision
import base64

warnings.filterwarnings("ignore")

st.set_page_config(page_title="تشخیص احساسات از گفتار", page_icon=":sparkles:")

def get_image_as_base64(image_file):
    with open(image_file, "rb") as image:
        return base64.b64encode(image.read()).decode()


# افزودن CSS برای بکگراند
image_base64 = get_image_as_base64('img.jpg')  # مسیر تصویر خود را اینجا وارد کنید
st.markdown(
    f"""
    <style>
        [data-testid="stSidebar"] > div:first-child {{
            background-img: url("data:image/jpg;base64, {image_base64}")
        }}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
    <style>
    .stButton > button {
        display: block;
        margin: 0 auto;
    }
    </style>
""", unsafe_allow_html=True)


st.title(':) در هر کلام تو، دنیایی از احساس نهفته است')

class Cnn_Lstm_Model(nn.Module):
    def __init__(self,num_emotions):
        super().__init__()
        
        self.num_emotions = num_emotions
        self.conv2Dblock = torchvision.models.resnet50()
        self.conv2Dblock.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.features = nn.Sequential(*(list(self.conv2Dblock.children())[:-1]))
        self.lstm = nn.LSTM(input_size=2048, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.LazyLinear(self.num_emotions)
        self.lstm_maxpool = nn.MaxPool2d(kernel_size=[2,4], stride=[2,4])
        
    def forward(self, x):
        
      y = self.features(x)
      y = y.flatten(2)
      y, _ = self.lstm(y.permute(0, 2, 1))
      y = self.fc(y)
      y = y.mean(dim=1)
      return y

model = torch.load('model_weight_balanced_aug.pt', map_location=torch.device('cpu'))
model.eval()

def record_audio(duration=5, fs=16000):
    st.markdown('<p style="text-align: center;">... ضبط صدا شروع شد</p>', unsafe_allow_html=True)
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    st.markdown('<p style="text-align: center;">... ضبط صدا تمام شد</p>', unsafe_allow_html=True)

    audio = audio.flatten()
    st.audio(audio, sample_rate=16000)
    audio = librosa.effects.preemphasis(audio)

    audio = librosa.util.normalize(audio)
    
    audio_tensor = torch.tensor(audio).unsqueeze(0)  

    mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=fs, n_fft=512, n_mels=128, hop_length=128)
    mel_specgram = mel_transform(audio_tensor)

    return audio, mel_specgram.unsqueeze(0)

if st.button('ضبط صدا'):
    with torch.no_grad():
        audio, mel = record_audio()
        output = model(mel)
        prob = torch.nn.functional.softmax(output, dim=1)
        feeling = torch.argmax(prob)
        match feeling:
            case 0:
                st.markdown('<h4 style="text-align: center;">😔 ای بابا حالا چرا ناراحت ؟</h4>', unsafe_allow_html=True)
            case 1:
                st.markdown('<h4 style="text-align: center;">😡 حالا چرا عصبانی ای ؟</h4>', unsafe_allow_html=True)
            case 2:
                st.markdown('<h4 style="text-align: center;">😀 ایول خیلی خوشحالی</h4>', unsafe_allow_html=True)
            case 3:
                st.markdown('<h4 style="text-align: center;">😐 !حالا انقدرم باهامون سرد نباش</h4>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p style="text-align: center;">طراحی شده با ❤️ توسط امیرحسین محجوب خراسانی</p>', unsafe_allow_html=True)
    