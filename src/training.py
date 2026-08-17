import os
import csv
import numpy as np
from collections import Counter

# Set variabel lingkungan sebelum meng-import tinygrad
# Menggunakan DEV=CPU sesuai dengan standar API tinygrad terbaru
os.environ["DEV"] = "CPU"

from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_save, safe_load

# ==========================================
# 1. BACA DATASET DARI CSV
# ==========================================
CSV_PATH = os.path.join("data_file", "data.csv")
MODEL_PATH = os.path.join("data_file", "model_weights.safetensors")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"File {CSV_PATH} tidak ditemukan! Jalankan data_random.py terlebih dahulu.")

raw_samples = []
with open(CSV_PATH, mode="r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader)  # Lewati header (angka_1, angka_2, angka_3)
    for row in reader:
        if row:
            raw_samples.append([int(x) for x in row])

raw_data = np.array(raw_samples)  # Shape: (N, 3)
num_samples = len(raw_data)
print(f"Berhasil memuat {num_samples} sampel data dari {CSV_PATH}.")

# ==========================================
# 2. PERSIAPAN DATASET UNTUK TINYGRAD
# ==========================================
NUM_CLASSES = 6
SEQ_LEN = 2  # Gunakan 2 baris sampel sebelumnya untuk memprediksi 1 baris sampel berikutnya

def to_one_hot(num_array):
    return np.eye(NUM_CLASSES)[num_array - 1]

X_raw, y_raw = [], []
for i in range(len(raw_data) - SEQ_LEN):
    seq_input = raw_data[i : i + SEQ_LEN]  # shape: (2, 3)
    target = raw_data[i + SEQ_LEN]         # shape: (3,)
    
    X_raw.append(to_one_hot(seq_input).reshape(-1))
    y_raw.append(to_one_hot(target))

X_train = np.array(X_raw, dtype=np.float32)  # Shape: (N-2, 36)
y_train = np.array(y_raw, dtype=np.float32)  # Shape: (N-2, 3, 6)

X_tensor = Tensor(X_train)
y_tensor = Tensor(y_train)

# ==========================================
# 3. DEFINISI MODEL NEURAL NETWORK
# ==========================================
INPUT_DIM = SEQ_LEN * 3 * NUM_CLASSES  # 2 * 3 * 6 = 36
HIDDEN_DIM = 64
OUTPUT_DIM = 3 * NUM_CLASSES          # 3 * 6 = 18

class DicePredictor:
    def __init__(self):
        self.w1 = Tensor.glorot_uniform(INPUT_DIM, HIDDEN_DIM)
        self.b1 = Tensor.zeros(HIDDEN_DIM)
        self.w2 = Tensor.glorot_uniform(HIDDEN_DIM, OUTPUT_DIM)
        self.b2 = Tensor.zeros(OUTPUT_DIM)

    def __call__(self, x):
        x = x.dot(self.w1).add(self.b1).relu()
        return x.dot(self.w2).add(self.b2)

model = DicePredictor()

# ==========================================
# 4. MEMUAT BOBOT (LOAD WEIGHTS) JIKA ADA
# ==========================================
if os.path.exists(MODEL_PATH):
    print(f"\nDitemukan file bobot tersimpan: '{MODEL_PATH}'")
    choice = input("Apakah ingin memuat bobot yang ada tanpa pelatihan ulang? (y/n): ").strip().lower()
    if choice == 'y':
        state_dict = safe_load(MODEL_PATH)
        load_state_dict(model, state_dict)
        print("-> Bobot berhasil dimuat ke dalam model!")
        skip_training = True
    else:
        skip_training = False
else:
    skip_training = False

# ==========================================
# 5. PROSES PELATIHAN (TRAINING LOOP)
# ==========================================
if not skip_training:
    Tensor.training = True

    optimizer = Adam([model.w1, model.b1, model.w2, model.b2], lr=0.01)
    EPOCHS = 100
    BATCH_SIZE = 16

    print("\nMemulai pelatihan tinygrad...")
    for epoch in range(EPOCHS):
        for i in range(0, len(X_train), BATCH_SIZE):
            x_batch = X_tensor[i : i + BATCH_SIZE]
            y_batch = y_tensor[i : i + BATCH_SIZE].reshape(-1, 3, NUM_CLASSES)

            logits = model(x_batch).reshape(-1, 3, NUM_CLASSES)
            
            probs = logits.softmax(axis=2)
            loss = -(y_batch * probs.log()).sum(axis=2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss.numpy():.4f}")

    # SIMPAN BOBOT MODEL (SAVE WEIGHTS)
    state_dict = get_state_dict(model)
    safe_save(state_dict, MODEL_PATH)
    print(f"\n-> Bobot model berhasil disimpan ke: '{MODEL_PATH}'")

Tensor.training = False

# ==========================================
# 6. ANALISIS FREKUENSI HISTORIS & PREDIKSI
# ==========================================
all_numbers = raw_data.flatten()
counter = Counter(all_numbers)

print("\n--- STATISTIK FREKUENSI DATASET ---")
for num in range(1, 7):
    freq = counter[num]
    print(f"Angka {num}: {freq} kali ({freq/len(all_numbers)*100:.2f}%)")

# Prediksi untuk baris berikutnya berdasarkan 2 baris terakhir di dataset
last_input = raw_data[-SEQ_LEN:]
input_onehot = to_one_hot(last_input).reshape(1, -1).astype(np.float32)
input_tensor = Tensor(input_onehot)

pred_logits = model(input_tensor).reshape(3, NUM_CLASSES)
pred_probs = pred_logits.softmax(axis=1).numpy()

print("\n--- PREDIKSI KOMBINASI SELANJUTNYA ---")
print(f"Berdasarkan 2 sampel terakhir:\n{last_input}\n")

predicted_numbers = []
for idx in range(3):
    probs = pred_probs[idx]
    predicted_digit = np.argmax(probs) + 1  # Kembalikan ke rentang 1-6
    predicted_numbers.append(predicted_digit)
    
    print(f"Posisi Angka ke-{idx+1}:")
    for digit in range(1, 7):
        print(f"  - Angka {digit}: {probs[digit-1]*100:.2f}%")
    print(f"  => Prediksi: {predicted_digit} ({probs[predicted_digit-1]*100:.2f}%)\n")

print(f">> Hasil Prediksi 3 Angka Selanjutnya: {tuple(predicted_numbers)}")
