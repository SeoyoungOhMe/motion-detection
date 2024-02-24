import os

def count_video_files(folder_path):
    # 지원하는 영상 파일 확장자 목록
    video_extensions = ['.mp4', '.avi', '.mov']
    video_count = 0

    # 주어진 폴더 경로 내의 모든 파일 및 폴더에 대해 반복
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 파일 확장자가 영상 파일 확장자 목록에 있는지 확인
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_count += 1

    return video_count

# 폴더 경로 지정 (이 경로를 실제 영상 파일이 있는 폴더 경로로 변경하세요)
folder_path = 'test_video\\final_sleeping_splitted'
print(f"{count_video_files(folder_path)}")
