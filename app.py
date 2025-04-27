import os
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from flask import Flask, request, jsonify
from keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from flask_cors import CORS
import noisereduce as nr

app = Flask(__name__)
CORS(app)

# 경로 설정
AUDIO_DIR = "./audio"
DATASET_DIR = "./multiclass_dataset"
MODEL_DIR = "./models"
MODEL_PATH = os.path.join(MODEL_DIR, "multiclass_voice_model.h5")
LABEL_CLASSES_PATH = os.path.join(MODEL_DIR, "label_classes.npy")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# MFCC 추출
def extract_mfcc(file_path, n_mfcc=13):
    y, sr = sf.read(file_path)
    y, _ = librosa.effects.trim(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

# 노이즈 제거
def reduce_noise_from_audio(filename):
    y, sr = librosa.load(filename, sr=16000)
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    sf.write(filename, reduced_noise, sr)
    print(f"Noise reduction applied to {filename}.")

# 모델 생성
def create_model(input_shape, num_classes):
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.SimpleRNN(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# 전체 데이터로 재학습
def train_model():
    X, y = [], []
    for user_folder in os.listdir(DATASET_DIR):
        user_path = os.path.join(DATASET_DIR, user_folder)
        if os.path.isdir(user_path):
            for file in os.listdir(user_path):
                if file.endswith('.wav'):
                    file_path = os.path.join(user_path, file)
                    features = extract_mfcc(file_path)
                    X.append(features)
                    y.append(user_folder)

    if not X:
        return False

    X = np.array(X)
    y = np.array(y)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = create_model(input_shape=(X.shape[1], 1), num_classes=len(le.classes_))
    model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

    model.save(MODEL_PATH)
    np.save(LABEL_CLASSES_PATH, le.classes_)
    print("Model training completed.")
    return True

# 회원가입 API
@app.route("/register", methods=["POST"])
def register():
    phone_number = request.form["phoneNumber"]
    audio_files = request.files.getlist("audio")

    if len(audio_files) != 3:
        return jsonify({"error": "3개의 음성 파일을 업로드해야 합니다."}), 400

    user_dir = os.path.join(DATASET_DIR, phone_number)
    os.makedirs(user_dir, exist_ok=True)

    saved_file_paths = []

    for i, file in enumerate(audio_files):
        file_path = os.path.join(user_dir, f"{phone_number}_{i+1}.wav")
        file.save(file_path)
        reduce_noise_from_audio(file_path)
        saved_file_paths.append(file_path)

    # 전체 데이터로 모델 재학습
    if train_model():
        # 모델 학습 성공 후 음성 파일 삭제
        for path in saved_file_paths:
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(user_dir) and not os.listdir(user_dir):
            os.rmdir(user_dir)

        return jsonify({"message": f"{phone_number}의 음성이 저장되고 모델이 재학습된 후 파일이 삭제되었습니다."})
    else:
        return jsonify({"error": "데이터가 부족하여 모델을 학습할 수 없습니다."}), 500

# 로그인 API
@app.route("/login", methods=["POST"])
def login():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_CLASSES_PATH):
        return jsonify({"error": "모델이 존재하지 않습니다. 먼저 회원가입을 해주세요."}), 400

    file = request.files["audio"]
    temp_path = os.path.join(AUDIO_DIR, "temp_login.wav")
    file.save(temp_path)
    reduce_noise_from_audio(temp_path)

    model = tf.keras.models.load_model(MODEL_PATH)
    classes = np.load(LABEL_CLASSES_PATH)

    features = extract_mfcc(temp_path).reshape(1, -1, 1)
    prediction = model.predict(features)
    predicted_phone_number = classes[np.argmax(prediction)]

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return predicted_phone_number  # 바로 phoneNumber 반환!

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)