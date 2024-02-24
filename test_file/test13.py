import cv2
import os
import time

def capture_frames(output_folder, duration):
    # 카메라 열기
    cap = cv2.VideoCapture(0)  # 0은 기본 카메라를 나타냄

    start_time = time.time()
    current_time = start_time

    while current_time - start_time <= duration:
        # 현재 시간 계산
        current_time = time.time()

        # 프레임 읽기
        ret, frame = cap.read()

        # 프레임 파일 경로 생성
        frame_path = os.path.join(output_folder, f"{int(current_time - start_time)}s.jpg")

        # 프레임 저장
        cv2.imwrite(frame_path, frame)

        # 1초마다 메시지 출력
        if int(current_time - start_time) % 1 == 0:
            print(f"캡처된 프레임: {int(current_time - start_time)}초")

    # 카메라 닫기
    cap.release()

if __name__ == "__main__":
    output_folder = "frames-1"
    duration = 10  # 캡처할 총 시간 (초)

    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    capture_frames(output_folder, duration)
