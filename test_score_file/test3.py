## 선아 언니 코드를 수정하여 대각선으로 나눈 값 매번 출력함.

import cv2
import mediapipe as mp
import numpy as np
import time
from math import sqrt

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

video_path = "C:\\motionDetection\\test_video\\aaa.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("동영상을 열 수 없습니다.")
    exit()

with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    
    start_time = time.time()  # 시작 시간 기록
    elapsed_time = 0  # 경과 시간 초기화
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("동영상이 끝났거나 찾을 수 없습니다.")
            continue
        
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        current_time = time.time()
        if current_time - start_time >= 1:
            start_time = current_time
            elapsed_time += 1
            
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
                    # 프레임의 대각선 길이 계산
                    frame_diagonal = sqrt(w ** 2 + h ** 2)
                    # final movement 계산
                    final_movement = avg_change / frame_diagonal
                    #print(f"{elapsed_time}sec - avg movement : {avg_change:.2f}, final movement : {final_movement:.4f}")
                    print(f"{elapsed_time}sec - final movement : {final_movement:.4f}")

        
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
