## SleepingBabies 최종 코드

import cv2
import mediapipe as mp
import numpy as np
import os
from math import sqrt

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

folder_path = "C:\\motionDetection\\sleepingBabies"
final_movements = []
total_videos = 0
recognized_videos = 0

print("< Sleeping Babies >")

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.mp4', '.avi', '.mov')):
        video_path = os.path.join(folder_path, filename)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"{filename}: EOF or Video Not Found.")
            continue
        total_videos += 1

        video_final_movements = []  # 이 영상의 final movements

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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
                    landmarks_list = []
                    for landmark in results.pose_landmarks.landmark:
                        if landmark.visibility < 0.5:
                            continue
                        h, w, c = image.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        landmarks_list.append((cx, cy))
                    
                    if len(landmarks_list) > 0:
                        changes = np.sqrt(np.sum(np.square(np.diff(landmarks_list, axis=0)), axis=1))
                        avg_change = np.mean(changes)
                        frame_diagonal = sqrt(w ** 2 + h ** 2)
                        final_movement = avg_change / frame_diagonal
                        video_final_movements.append(final_movement)

            if video_final_movements:
                video_avg_final_movement = sum(video_final_movements) / len(video_final_movements)
                final_movements.append(video_avg_final_movement)
                recognized_videos += 1
                print(f"{filename} - avg final movement : {video_avg_final_movement:.4f}")
            else:
                print(f"{filename}: No recognizable movements found.")

        cap.release()

cv2.destroyAllWindows()

print(f"\n{recognized_videos}/{total_videos} recognized.")

# 전체 평균 final movement 계산 및 출력
if final_movements:
    avg_final_movement = sum(final_movements) / len(final_movements)
    print(f"\nAverage Final Movement across all sleeping baby videos: {avg_final_movement:.4f}")
else:
    print("\nNo final movement data available.")

