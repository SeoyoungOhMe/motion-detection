import cv2
import mediapipe as mp
import numpy as np
import time
from math import sqrt

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 동영상 파일을 엽니다.
# video_path = "C:/KIMSEONAH/Test_Study/MotionDetection/fly-ai-project-video-to-clip/원본동영상/gettyimages-82627946-640_adpp.mp4"
video_path = "C:\\motionDetection\\test_video\\aaa.mp4"
cap = cv2.VideoCapture(video_path)
# cap=cv2.VideoCapture(0)
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
            # break
            continue
        
        # 필요에 따라 성능 향상을 위해 이미지 작성을 불가능함으로 기본 설정
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
        
        # 1초마다 좌표를 출력하고 변화량을 계산
        current_time = time.time()
        if current_time - start_time >= 1:
            start_time = current_time  # 시작 시간 갱신
            elapsed_time += 1  # 경과 시간 증가
            
            if results.pose_landmarks:
                landmarks_list = []
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    if landmark.visibility < 0.5:  # 감지하지 못한 값을 처리합니다.
                        continue
                    h, w, c = image.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    landmarks_list.append((cx, cy))
                
                if len(landmarks_list) > 0:
                    changes = np.sqrt(np.sum(np.square(np.diff(landmarks_list, axis=0)), axis=1)) # 이전 좌표와 현재 좌표 사이의 변화량 계산
                    avg_change = np.mean(changes) # 변화량의 평균값 계산
                    print(f"{elapsed_time}sec - avg movement : {avg_change:.2f}")  # 초와 평균 변화량 출력
        
        # 이미지를 그대로 표시합니다.
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()