## 매초의 final value 출력하지만 0sec으로만 출력되고 최종 평균값 구해주지 않음

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

print("< Sleeping Babies >")

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.mp4', '.avi', '.mov')):
        video_path = os.path.join(folder_path, filename)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"{filename}: EOF or Video Not Found.")
            continue

        print(f"Processing {filename}...")

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            start_time = time.time()
            elapsed_time = -1
            prev_landmarks_list = []
            total_distance = 0
            cnt = 0
            
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                current_time = time.time()
                elapsed_time = current_time - start_time

                curr_landmarks_list = []

                if elapsed_time >= 1:
                    start_time = current_time
                    elapsed_time = 0

                    if results.pose_landmarks:
                        for idx, landmark in enumerate(results.pose_landmarks.landmark):
                            if idx in [0, 15, 16, 17, 18, 19, 20, 27, 28]:
                                h, w, c = image.shape
                                cx, cy = int(landmark.x * w), int(landmark.y * h)
                                if landmark.visibility < 0.5:
                                    cx, cy = 0, 0
                                curr_landmarks_list.append((cx, cy))
                        
                        total_distance = 0
                        cnt = 0
                        
                        if prev_landmarks_list and curr_landmarks_list:
                            for prev_landmark, curr_landmark in zip(prev_landmarks_list, curr_landmarks_list):
                                if prev_landmark == (0, 0) or curr_landmark == (0, 0):
                                    continue
                                distance = sqrt((curr_landmark[0] - prev_landmark[0])**2 + (curr_landmark[1] - prev_landmark[1])**2)
                                total_distance += distance
                                cnt += 1
                        
                        if cnt > 0:
                            diagonal_length = sqrt(w**2 + h**2) * cnt
                            final_value = total_distance / diagonal_length
                            print(f"{filename} - {int(time.time() - start_time)} sec - Final Value: {final_value:.4f}")
                        else:
                            print(f"{filename} - {int(time.time() - start_time)} sec - No movement detected.")

                        prev_landmarks_list = curr_landmarks_list.copy()

                cv2.imshow('MediaPipe Pose', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break

        cap.release()

cv2.destroyAllWindows()
