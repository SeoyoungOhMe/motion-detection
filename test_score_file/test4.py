## test3.py(인풋이 영상 하나)를 SleepingBabies의 모든 영상이 되도록 수정.
## 인식된 동영상의 갯수와 최종 평균 값만 출력

import cv2
import mediapipe as mp
import numpy as np
import time
from math import sqrt
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

folder_path = "C:\\motionDetection\\SleepingBabies"
final_movements = []  # final movement 값을 저장할 리스트
total_videos = 0  # 전체 영상 수
recognized_videos = 0  # 인식된 영상 수

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.mp4', '.avi', '.mov')):
        video_path = os.path.join(folder_path, filename)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"동영상을 열 수 없습니다: {filename}")
            continue
        total_videos += 1

        with mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as pose:
            
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
                        final_movements.append(final_movement)

            if len(final_movements) > 0:  # 이 영상에서 최소 한 번은 인식됨
                recognized_videos += 1

        cap.release()

cv2.destroyAllWindows()

# print(f"Total videos processed: {total_videos}")
# print(f"Videos with recognized movements: {recognized_videos}")
print(f"{recognized_videos}/{total_videos} recognized.")

# 평균 final movement 계산 및 출력
if final_movements:
    avg_final_movement = sum(final_movements) / len(final_movements)
    print(f"Average Final Movement: {avg_final_movement:.4f}")
else:
    print("No final movement data available.")


