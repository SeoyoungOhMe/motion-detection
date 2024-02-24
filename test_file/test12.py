# 실패!

import cv2
import os
import time

def capture_frames(output_folder, duration):
    # 카메라 열기
    cap = cv2.VideoCapture(0)  # 0은 기본 카메라를 나타냄

    start_time = time.time()

    previous_frame_path = None

    while True:
        # 현재 시간 계산
        current_time = time.time()

        # 프레임 읽기
        ret, frame = cap.read()

        # 현재 시간과 시작 시간 간의 차이 계산
        elapsed_time = current_time - start_time

        # 지정된 기간 동안만 프레임 저장
        if elapsed_time <= duration:
            # 프레임 파일 경로 생성
            frame_path = os.path.join(output_folder, f"{int(elapsed_time)}s.jpg")

            # 프레임 저장
            cv2.imwrite(frame_path, frame)

            # 이전 프레임 삭제
            if elapsed_time >= 2 and previous_frame_path is not None:
                previous_frame_path = os.path.join(output_folder, f"{int(elapsed_time) - 2}s.jpg")
                os.remove(previous_frame_path)

            # 이전 프레임 경로 업데이트
            previous_frame_path = frame_path
        else:
            break

    # 카메라 닫기
    cap.release()

if __name__ == "__main__":
    output_folder = "captured_frames-7"
    duration = 20  # 캡처할 총 시간 (초)

    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    capture_frames(output_folder, duration)
