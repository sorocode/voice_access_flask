import os

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
import tensorflow as tf
from flask import Flask, request, jsonify
from keras import layers
from keras import models
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
AUDIO_DIR = "./audio"
MODEL_DIR = "./models"
LOGIN_COUNT_FILE = os.path.join(AUDIO_DIR, "login_count.txt")

# 노이즈 제거 함수
def reduce_noise_from_audio(filename):
    y, sr = librosa.load(filename, sr=16000)
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    sf.write(filename, reduced_noise, sr)
    print(f"Noise reduction applied to {filename}.")

# FLAC 파일에서 MFCC 특징 추출 함수
def extract_mfcc_features_from_flac(audio_file):
    y, sr = sf.read(audio_file)
    y, _ = librosa.effects.trim(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# RNN 모델 생성 함수
def create_rnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.SimpleRNN(64, activation='relu', return_sequences=False))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 회원가입 API
@app.route("/register", methods=["POST"])
def register_user():
    phone_number = request.form["phoneNumber"]
    audio_files = request.files.getlist("audio")
    user_dir = os.path.join(MODEL_DIR, phone_number)
    os.makedirs(user_dir, exist_ok=True)

    features_list = []
    temp_file_paths = []  # 삭제할 파일 경로 저장용

    for i, file in enumerate(audio_files):
        file_path = os.path.join(AUDIO_DIR, f"{phone_number}_register_{i+1}.flac")
        file.save(file_path)
        temp_file_paths.append(file_path)
        reduce_noise_from_audio(file_path)
        features = extract_mfcc_features_from_flac(file_path)
        features_list.append(features)

    features = np.array(features_list).reshape(len(features_list), -1, 1)
    model = create_rnn_model((features.shape[1], 1))
    model.fit(features, np.ones(len(features)), epochs=10)
    model.save(os.path.join(user_dir, "voice_authentication_model.h5"))

    # 임시 파일 삭제
    for path in temp_file_paths:
        if os.path.exists(path):
            os.remove(path)

    return jsonify({"message": f"{phone_number}'s VOICE MODEL was saved successfully."})

# 로그인 API
@app.route("/login", methods=["POST"])
def login():
    try:
        with open(LOGIN_COUNT_FILE, 'r') as f:
            login_count = int(f.read().strip())
    except (FileNotFoundError, ValueError):
        login_count = 0
        # 파일이 없거나 읽을 수 없는 경우, 새로 생성하고 0을 씀
        with open(LOGIN_COUNT_FILE, 'w') as f:
            f.write(str(login_count))

    login_count += 1
    with open(LOGIN_COUNT_FILE, 'w') as f:
        f.write(str(login_count))

    file = request.files["audio"]
    file_path = os.path.join(AUDIO_DIR, f"login{login_count}.wav")
    file.save(file_path)

    try:
        reduce_noise_from_audio(file_path)
        input_features = extract_mfcc_features_from_flac(file_path)
        if input_features is None or np.all(input_features == 0):
            return jsonify({"message": "Invalid audio input."}), 400
        input_features = input_features.reshape((1, input_features.shape[0], 1))

        for user_folder in os.listdir(MODEL_DIR):
            model_path = os.path.join(MODEL_DIR, user_folder, "voice_authentication_model.h5")
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                prediction = model.predict(input_features)
                if prediction >= 0.8:
                    print(f"prediction: {prediction}. {user_folder}: LOGIN SUCCESS")
                    return user_folder

        return jsonify({"message": "LOGIN FAILED"})
    finally:
        # 처리 후 음성 파일 삭제
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
