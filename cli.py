import librosa
import numpy as np
import pyaudio
import wave
import os
import tensorflow as tf
import tensorflow_hub as hub
from keras import layers
from keras import models

import pickle
import soundfile as sf  # FLAC 파일을 읽기 위한 라이브러리 추가
import noisereduce as nr  # 노이즈 제거 라이브러리

# 전역 경로 상수
AUDIO_FOLDER = "./audio"
MODEL_FOLDER = "./models"

# FLAC 파일에서 MFCC 특징 추출 함수
def extract_mfcc_features_from_flac(audio_file):
    y, sr = sf.read(audio_file)
    y, _ = librosa.effects.trim(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# RNN 모델 정의
def create_rnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.SimpleRNN(64, activation='relu', return_sequences=False))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 사용자 음성 녹음 함수
def record_audio(filename, duration=5):
    p = pyaudio.PyAudio()
    format = pyaudio.paInt16
    channels = 1
    rate = 16000
    frames_per_buffer = 1024

    stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=frames_per_buffer)
    frames = [stream.read(frames_per_buffer) for _ in range(0, int(rate / frames_per_buffer * duration))]

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

    reduce_noise_from_audio(filename)

# 노이즈 제거 함수
def reduce_noise_from_audio(filename):
    y, sr = librosa.load(filename, sr=16000)
    sf.write(filename, nr.reduce_noise(y=y, sr=sr), sr)

# 사용자 모델 저장 함수
def save_user_model(username, audio_file):
    model_path = os.path.join(AUDIO_FOLDER, username, f"{username}_model.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, 'wb') as f:
        pickle.dump({'username': username, 'features': extract_mfcc_features_from_flac(audio_file)}, f)

# 로그인 인증 함수
def authenticate_for_login_with_rnn():
    count_file = os.path.join(AUDIO_FOLDER, "login_count.txt")
    login_count = int(open(count_file).read().strip()) + 1 if os.path.exists(count_file) else 1

    audio_file_to_authenticate = os.path.join(AUDIO_FOLDER, f"login{login_count}.wav")
    print(f"{login_count}번째 로그인 시도 중입니다. 말씀하세요.")
    record_audio(audio_file_to_authenticate, duration=5)
    open(count_file, 'w').write(str(login_count))

    username = authenticate_user_with_rnn(audio_file_to_authenticate)
    print(f"{username}님의 로그인 성공!" if username else "로그인 실패")

# 음성 기반 사용자 인증
def authenticate_user_with_rnn(audio_file):
    input_features = extract_mfcc_features_from_flac(audio_file)
    if input_features is None or np.all(input_features == 0):
        return None

    input_features = input_features.reshape((1, input_features.shape[0], 1))
    for user_folder in os.listdir(MODEL_FOLDER):
        user_model_path = os.path.join(MODEL_FOLDER, user_folder, 'voice_authentication_model.h5')
        if os.path.exists(user_model_path):
            model = tf.keras.models.load_model(user_model_path)
            prediction = model.predict(input_features)
            if prediction >= 0.8:
                return user_folder
    return None

# 사용자 등록 함수
def register_user(username):
    user_audio_folder = os.path.join(AUDIO_FOLDER, username)
    if os.path.exists(user_audio_folder):
        print(f"User '{username}' is already registered.")
        return
    os.makedirs(user_audio_folder, exist_ok=True)

    audio_files = [os.path.join(user_audio_folder, f"{username}_register_{i+1}.flac") for i in range(5)]
    print(f"{username}님의 음성을 5번 녹음해주세요.")

    features_list = [extract_mfcc_features_from_flac(record_audio(file, duration=7)) for file in audio_files]
    features = np.array(features_list).reshape((len(features_list), features_list[0].shape[0], 1))

    model = create_rnn_model(input_shape=(features.shape[1], 1))
    model.fit(features, np.array([1] * len(features)), epochs=10)

    user_model_folder = os.path.join(MODEL_FOLDER, username)
    os.makedirs(user_model_folder, exist_ok=True)
    model.save(os.path.join(user_model_folder, "voice_authentication_model.h5"))
    print(f"{username}님의 음성 모델이 저장되었습니다.")

# 메인 메뉴
def main():
    while True:
        choice = input("\n1. 로그인\n2. 회원가입\n0. 종료\n선택 (1/2/0): ")
        if choice == '1':
            authenticate_for_login_with_rnn()
        elif choice == '2':
            register_user(input("회원가입할 사용자 이름을 입력하세요: "))
        elif choice == '0':
            break
        else:
            print("잘못된 선택입니다.")

if __name__ == "__main__":
    main()