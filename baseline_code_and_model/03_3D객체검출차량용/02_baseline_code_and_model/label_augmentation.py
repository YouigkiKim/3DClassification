import os

# 디렉터리 경로
input_dir = "/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/labels"
output_dir = "/home/ailab/git/Team_3/baseline_code_and_model/03_3D객체검출차량용/02_baseline_code_and_model/labels"

# 중복 횟수 설정
PEDESTRIAN_CLASS = "Pedestrian"
CYCLIST_CLASS = "Cyclist"
PEDESTRIAN_DUPLICATION = 3
CYCLIST_DUPLICATION = 5

# 라벨을 처리하고 중복하는 함수
def duplicate_labels(input_file, output_file):
    with open(input_file, "r") as f:
        lines = f.readlines()

    new_lines = []  # 새로운 라벨을 저장할 리스트

    for line in lines:
        elements = line.strip().split()
        
        # 8번째 요소 (index 7)가 pedestrian인지 cyclist인지 확인
        if elements[7] == PEDESTRIAN_CLASS:
            # pedestrian 라벨을 3번 추가
            new_lines.extend([line] * PEDESTRIAN_DUPLICATION)
        elif elements[7] == CYCLIST_CLASS:
            # cyclist 라벨을 5번 추가
            new_lines.extend([line] * CYCLIST_DUPLICATION)
        else:
            # 다른 라벨은 그대로 추가
            new_lines.append(line)

    # 새로운 라벨 파일에 저장
    with open(output_file, "w") as f:
        f.writelines(new_lines)

    print(f"새로운 라벨 파일이 생성되었습니다: {output_file}")

# 모든 txt 파일에 대해 함수 호출
for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
        duplicate_labels(input_file, output_file)
