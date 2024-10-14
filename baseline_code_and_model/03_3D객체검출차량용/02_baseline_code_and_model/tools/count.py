import numpy as np
import os

# 00000000 ~ 00004937 숫자 범위 설정
start_num = 0
end_num = 4937

# 숫자 리스트 생성
numbers = [f"{i:08d}" for i in range(start_num, end_num + 1)]

# 8:2 비율로 섞어서 나누기
np.random.shuffle(numbers)
split_idx = int(len(numbers) * 0.8)
train_numbers = numbers[:split_idx]
val_numbers = numbers[split_idx:]

# 저장 경로 설정
train_txt_path = "/home/ailab/git/Team_3/baseline_code_and_model/03_3D객체검출차량용/02_baseline_code_and_model/data/custom_av/ImageSets/train.txt"
val_txt_path = "/home/ailab/git/Team_3/baseline_code_and_model/03_3D객체검출차량용/02_baseline_code_and_model/data/custom_av/ImageSets/val.txt"

# train.txt 파일 저장
with open(train_txt_path, 'w') as train_file:
    for num in train_numbers:
        train_file.write(f"{num}\n")

# val.txt 파일 저장
with open(val_txt_path, 'w') as val_file:
    for num in val_numbers:
        val_file.write(f"{num}\n")

print(f"Train and validation sets have been successfully saved.")
