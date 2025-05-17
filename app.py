import os
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from flask import Flask, request, jsonify
from flask_cors import CORS
from keras.models import load_model
# 프로젝트 디렉토리 설정
AUDIO_DIR = "./audio"
MODEL_DIR = "./models"
DATASET_DIR = "./multiclass_dataset"
MODEL_PATH = os.path.join(MODEL_DIR, "multiclass_voice_model.keras")
LABEL_PATH = os.path.join(MODEL_DIR, "label_classes.npy")
LOGIN_COUNT_FILE = os.path.join(AUDIO_DIR, "login_count.txt")

# Flask 앱 초기화
app = Flask(__name__)
CORS(app)

def reduce_noise_from_audio(filename):
    y, sr = librosa.load(filename, sr=16000)
    reduced = nr.reduce_noise(y=y, sr=sr)
    sf.write(filename, reduced, sr)

def extract_mfcc(file_path, n_mfcc=13, fixed_length=200):
    y, sr = sf.read(file_path)
    y, _ = librosa.effects.trim(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    combined = np.vstack([mfcc, delta, delta2])
    mean = np.mean(combined, axis=1, keepdims=True)
    std = np.std(combined, axis=1, keepdims=True) + 1e-6
    combined = (combined - mean) / std

    if combined.shape[1] < fixed_length:
        pad_width = fixed_length - combined.shape[1]
        combined = np.pad(combined, ((0, 0), (0, pad_width)), mode='constant')
    else:
        combined = combined[:, :fixed_length]

    return combined.T  # shape: (time, 39)

def is_silent(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    trimmed, _ = librosa.effects.trim(y, top_db=40)
    return len(trimmed) == 0

# 회원가입 API
@app.route("/register", methods=["POST"])
def register_user():
    phone_number = request.form["phoneNumber"]
    audio_files = request.files.getlist("audio")
    user_dir = os.path.join(DATASET_DIR, phone_number)

    if os.path.exists(user_dir):
        return jsonify({"message": "이미 등록된 사용자입니다."}), 400

    os.makedirs(user_dir, exist_ok=True)

    for i, file in enumerate(audio_files[:5]):
        file_path = os.path.join(user_dir, f"{phone_number}_{i+1}.wav")
        file.save(file_path)
        reduce_noise_from_audio(file_path)

    # 데이터 증강
    for i in range(5):
        orig_path = os.path.join(user_dir, f"{phone_number}_{i+1}.wav")
        y, sr = librosa.load(orig_path, sr=None)
        aug1 = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
        aug2 = librosa.effects.time_stretch(y, rate=1.2)
        aug3 = y + np.random.randn(len(y)) * 0.01
        for j, aug in enumerate([aug1, aug2, aug3]):
            aug_path = os.path.join(user_dir, f"{phone_number}_{i+1}_aug_{j+1}.wav")
            sf.write(aug_path, aug, sr)

    # 모델 재학습
    train_model()

    # TODO: 사용자의 원본 음성 데이터 삭제

    return jsonify({"message": f"{phone_number} 등록 완료 및 모델 재학습 완료"})

#  로그인 API
@app.route("/login", methods=["POST"])
def login():
    print("📥 request.headers:", request.headers)
    print("📥 request.files:", request.files)
    print("📥 request.form:", request.form)

    if "audio" not in request.files:
        return jsonify({"message": "audio 파트가 없습니다"}), 400
    try:
        with open(LOGIN_COUNT_FILE, 'r') as f:
            login_count = int(f.read().strip())
    except:
        login_count = 0
    login_count += 1
    with open(LOGIN_COUNT_FILE, 'w') as f:
        f.write(str(login_count))

    file = request.files["audio"]
    path = os.path.join(AUDIO_DIR, f"login{login_count}.wav")
    file.save(path)
    reduce_noise_from_audio(path)
    print(f"📄 파일 저장 경로: {path}")
    print(f"📏 파일 크기: {os.path.getsize(path)} bytes")

    if is_silent(path):
        print("🚫 무음으로 판단됨.")
        os.remove(path)
        return jsonify({"message": "무음입니다."}), 400

    model = load_model(MODEL_PATH)
    classes = np.load(LABEL_PATH, allow_pickle=True)
    feature = extract_mfcc(path).reshape(1, 200, 39)
    pred = model.predict(feature)
    idx = np.argmax(pred)
    phone_number = classes[idx]
    conf = float(pred[0][idx])

    os.remove(path)

    if conf < 0.5 or phone_number.lower() == "unknown":
        return jsonify({"message": "로그인 실패", "confidence": conf})
    print(f"phoneNumber:  {phone_number}")
    return phone_number

#  학습 함수
def train_model():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from keras.models import Sequential
    from keras.layers import Conv1D, MaxPooling1D, Dropout, Bidirectional, LSTM, GlobalAveragePooling1D, Dense
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau

    X, y = [], []
    for user in os.listdir(DATASET_DIR):
        user_dir = os.path.join(DATASET_DIR, user)
        for file in os.listdir(user_dir):
            if file.endswith(".wav"):
                fpath = os.path.join(user_dir, file)
                feat = extract_mfcc(fpath)
                X.append(feat)
                y.append(user)

    X = np.array(X)
    y = np.array(y)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    classes = le.classes_

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = Sequential([
        Conv1D(128, 5, activation='relu', input_shape=(200, 39)),
        MaxPooling1D(2), Dropout(0.3),
        Conv1D(128, 3, activation='relu'),
        MaxPooling1D(2), Dropout(0.3),
        Bidirectional(LSTM(128, return_sequences=True)), Dropout(0.3),
        GlobalAveragePooling1D(),
        Dense(128, activation='relu'), Dropout(0.3),
        Dense(len(classes), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(
        X_train, y_train,
        epochs=100,
        validation_data=(X_test, y_test),
        callbacks=[
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6)
        ]
    )

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    np.save(LABEL_PATH, classes)
    print("모델 학습 완료.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)