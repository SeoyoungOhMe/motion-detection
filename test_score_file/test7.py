## 선아언니ver 1차 수정

import cv2
import mediapipe as mp
import numpy as np
import time
from math import sqrt

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 동영상 파일을 엽니다.
video_path = "C:\\motionDetection\\test_video\\vod.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("동영상을 열 수 없습니다.")
    exit()
    
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
    
    start_time = time.time()  # 시작 시간 기록
    elapsed_time = -1  # 경과 시간 초기화
    
    prev_landmarks_list=[]
    total_distance=0
    cnt=0
    
    while cap.isOpened():
        success, image=cap.read()
        if not success:
            print("동영상 찾을 수 없음")
            break
        
        # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정합니다.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        # 포즈 주석을 이미지 위에 그립니다.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        # 현재시간과 시작 시간의 차이
        current_time = time.time()
        elapsed_time=current_time-start_time
        
        curr_landmarks_list=[]
        
        # 1초마다 좌표를 출력합니다.
        if elapsed_time >= 1:
            start_time = current_time  # 시작 시간 갱신
            elapsed_time =0  # 경과 시간 증가
            
            if results.pose_landmarks:
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    if idx in [0, 15, 16, 17, 18, 19, 20, 27, 28]:
                        h, w, c = image.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        if landmark.visibility < 0.5: #감지하지 못한 값은?
                            cx, cy=0,0
                        curr_landmarks_list.append((cx, cy)) 
                        # print(f"Coordinate of {idx} on {elapsed_time} sec : ({cx}, {cy})")  # 초와 점의 좌표 출력
                
                # print('prev_landmarks_list', prev_landmarks_list)
                # print('curr_landmarks_list', curr_landmarks_list)
                
                total_distance=0
                cnt=0
                if prev_landmarks_list and curr_landmarks_list:
                    prev_landmarks_array = np.array(prev_landmarks_list)
                    curr_landmarks_array = np.array(curr_landmarks_list)
                    
                    
                    for prev_landmark, curr_landmark in zip(prev_landmarks_list, curr_landmarks_list):
                        # 만약 어느 하나의 좌표가 (0, 0)인 경우 거리를 계산하지 않음
                        if prev_landmark == (0, 0) or curr_landmark == (0, 0):
                            continue
                         # 두 좌표 사이의 거리 계산
                        dx = curr_landmark[0] - prev_landmark[0]
                        dy = curr_landmark[1] - prev_landmark[1]
                        distance = sqrt(dx**2 + dy**2)
                        total_distance+=distance
                        cnt+=1
                        # print(cnt)
                    # print("Distance Difference:", total_distance, '\n')
                    # print(total_distance, "dfdf", cnt)
                    
                else:
                    print('0 sec\n')
                    
                prev_landmarks_list=curr_landmarks_list.copy()
                
                # 프레임의 대각선 길이 계산
                h, w, c = image.shape
                if cnt>0:
                    diagonal_length = sqrt(w**2 + h**2)*cnt
                    final_value = (total_distance/ diagonal_length)
                    print(f"Final Value: {final_value}")
                    # print("\n")
                    # print(f"Frame Diagonal Length: {diagonal_length}")
                else:
                    print('0 sec')

                        
                
        # 이미지를 그대로 표시합니다.
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
                        
cap.release()
cv2.destroyAllWindows()