## 매초마다 최대, 최소 랜드마크의 x, y 좌표 출력

import cv2
import mediapipe as mp
import numpy as np
import os
from math import sqrt
import time

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

folder_path = "C:\\motionDetection\\sleeping"  # 비디오 파일이 있는 폴더 경로
all_final_values = []  # 모든 영상에서 계산된 final value들을 저장할 리스트

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.mp4', '.avi', '.mov')):
        video_path = os.path.join(folder_path, filename)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"{filename}: EOF or Video Not Found.")
            continue

        print(f"\nProcessing {filename}...")
        
        second_count = -1  # 초를 세는 변수를 초기화
        min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0  # 매 초마다 최소, 최대 좌표를 저장할 변수 초기화

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            start_time = time.time()

            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    for idx, landmark in enumerate(results.pose_landmarks.landmark):
                        if idx in [0, 15, 16, 17, 18, 19, 20, 27, 28]:
                            landmark_x = int(landmark.x * image.shape[1])
                            landmark_y = int(landmark.y * image.shape[0])
                            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), -1)

                            # 매 초마다 최소, 최대 x, y 좌표 업데이트
                            min_x = min(min_x, landmark_x)
                            min_y = min(min_y, landmark_y)
                            max_x = max(max_x, landmark_x)
                            max_y = max(max_y, landmark_y)

                # 현재 처리 초를 계산합니다.
                if time.time() - start_time >= 1:
                    second_count += 1
                    start_time = time.time()

                    # 매 초마다 최소, 최대 좌표 출력
                    print(f"{second_count} sec - Min X: {min_x}, Min Y: {min_y}, Max X: {max_x}, Max Y: {max_y}")

                    # 변수 초기화
                    min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0

                cv2.imshow('MediaPipe Pose', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

            cap.release()

cv2.destroyAllWindows()

if all_final_values:
    avg_final_value = sum(all_final_values) / len(all_final_values)
    print(f"\nAverage Final Value across all videos: {avg_final_value:.4f}")
else:
    print("\nNo final movement data available.")
