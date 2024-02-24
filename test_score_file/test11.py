## 비디오에 아무것도 표시 안되고 매초 값 출력, 최종 평균도 출력

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

print("< Sleeping Babies >")
#print("< Not Sleeping Babies >")

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.mp4', '.avi', '.mov')):
        video_path = os.path.join(folder_path, filename)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"{filename}: EOF or Video Not Found.")
            continue

        print(f"\nProcessing {filename}...")

        second_count = -1  # 초를 세는 변수를 초기화하고 각 영상 처리 시작 시 0으로 설정

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            start_time = time.time()
            prev_landmarks_list = []

            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # 현재 처리 초를 계산합니다.
                if time.time() - start_time >= 1:
                    second_count += 1
                    start_time = time.time()

                    total_distance = 0
                    cnt = 0

                    if results.pose_landmarks:
                        curr_landmarks_list = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0]))
                                               for idx, landmark in enumerate(results.pose_landmarks.landmark)
                                               if idx in [0, 15, 16, 17, 18, 19, 20, 27, 28] and landmark.visibility > 0.5]

                        if prev_landmarks_list and curr_landmarks_list:
                            distances = [sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
                                         for prev, curr in zip(prev_landmarks_list, curr_landmarks_list)]
                            total_distance = sum(distances)
                            cnt = len(distances)

                        if cnt > 0:
                            diagonal_length = sqrt(image.shape[1]**2 + image.shape[0]**2) * cnt
                            final_value = total_distance / diagonal_length
                            all_final_values.append(final_value)
                            #print(f"{filename} - {second_count} sec - Final Value: {final_value:.4f}")
                            print(f"{second_count} sec - {final_value:.4f}")

                        prev_landmarks_list = curr_landmarks_list

                cv2.imshow('MediaPipe Pose', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

            cap.release()

cv2.destroyAllWindows()

# 모든 영상의 처리가 끝난 후 final value들의 평균을 계산하고 출력
if all_final_values:
    avg_final_value = sum(all_final_values) / len(all_final_values)
    print(f"\nAverage Final Value across all videos: {avg_final_value:.4f}")
else:
    print("\nNo final movement data available.")
