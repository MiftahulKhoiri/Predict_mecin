import os
import csv
import random

def get_next_filename(folder_name):
    # Cek nama file yang belum digunakan (data.csv, data1.csv, data2.csv, dst.)
    base_name = "data"
    extension = ".csv"
    
    first_file = os.path.join(folder_name, f"{base_name}{extension}")
    if not os.path.exists(first_file):
        return first_file
    
    counter = 1
    while True:
        next_file = os.path.join(folder_name, f"{base_name}{counter}{extension}")
        if not os.path.exists(next_file):
            return next_file
        counter += 1

def generate_sample_data():
    # 1. Minta input dari user berapa jumlah baris data yang ingin dibuat
    try:
        user_input = input("Masukkan jumlah baris data yang ingin dibuat (contoh: 100): ")
        num_rows = int(user_input)
    except ValueError:
        print("Input tidak valid! Harap masukkan angka bulat.")
        return

    # 2. Buat folder 'data_file' jika belum ada
    folder_name = "data_file"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # 3. Tentukan nama file berurutan agar tidak menimpa data lama
    file_path = get_next_filename(folder_name)

    # 4. Generate data acak dan simpan ke file CSV
    with open(file_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # Menulis header kolom
        writer.writerow(["angka_1", "angka_2", "angka_3"])

        # Menulis baris data acak
        for _ in range(num_rows):
            a1 = random.randint(1, 6)
            a2 = random.randint(1, 6)
            a3 = random.randint(1, 6)
            writer.writerow([a1, a2, a3])

    print(f"\nBerhasil membuat {num_rows} baris data sampel!")
    print(f"File tersimpan di: {os.path.abspath(file_path)}")

if __name__ == "__main__":
    generate_sample_data()
