## 최종 csv 폴더에 움직임 저장하는 코드

import os
import cv2
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pandas as pd
from math import sqrt
import time
import csv

output_csv_path = 'movement_scores.csv'

# Mediapipe 솔루션 초기화
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

base_options = python.BaseOptions(model_asset_path='C:\\motionDetection\\face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)


def write_movement_scores( input_folder_path ):
    
    all_final_values = []  # 모든 영상에서 계산된 final value들을 저장할 리스트
    
    # if not os.path.isfile(output_csv_path):
    #     print(f"CSV file {output_csv_path} not found.")
    #     return
    
    # pandas 데이터프레임으로 초기화
    scores_df = pd.DataFrame([ 'video_file','movement_score' ])

    # 'video_file' 열이 없으면 추가
    if 'video_file' not in scores_df.columns:
        scores_df['video_file'] = None
    
    if 'movement_score' not in scores_df.columns:
        scores_df['movement_score'] = None  # 새 열 추가, 초기값은 None
        
    for filename in os.listdir( input_folder_path ):
        if filename.lower().endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join( input_folder_path , filename)
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"{filename}: EOF or Video Not Found.")
                continue

            #print(f"\nProcessing {filename}...")
            
            second_count = -1  # 초를 세는 변수를 초기화
            prev_landmarks_list = []  # 이전 프레임의 랜드마크 리스트 초기화

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

                    curr_landmarks_list = []
                    recognized_landmarks_count = 0  # 인식된 랜드마크의 개수를 카운트할 변수 초기화
                    min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0  # 매 초마다 최소, 최대 좌표를 저장할 변수 초기화
                    
                    # 필요한 변수들을 초기화
                    center_points_sum_x = 0
                    center_points_sum_y = 0
                    center_points_count = 0
                    center_point_x_pixel = 0
                    center_point_y_pixel = 0

                    if results.pose_landmarks:
                        for idx, landmark in enumerate(results.pose_landmarks.landmark):
                            
                            if idx in [11, 12, 23, 24]:
                                center_points_sum_x += landmark.x
                                center_points_sum_y += landmark.y
                                center_points_count += 1
                            
                            if center_points_count > 0:
                                center_point_x = center_points_sum_x / center_points_count
                                center_point_y = center_points_sum_y / center_points_count
                                # 이미지 좌표로 변환
                                center_point_x_pixel = int(center_point_x * image.shape[1])
                                center_point_y_pixel = int(center_point_y * image.shape[0])
                                
                            if idx in [0, 15, 16, 17, 18, 19, 20, 27, 28]:
                                recognized_landmarks_count += 1  # 인식된 랜드마크 카운트
                                landmark_x = int(landmark.x * image.shape[1]) - center_point_x_pixel
                                landmark_y = int(landmark.y * image.shape[0]) - center_point_y_pixel
                                # cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), -1)
                                curr_landmarks_list.append((landmark_x, landmark_y))

                                # 최소, 최대 x, y 좌표 업데이트
                                min_x = min(min_x, landmark_x)
                                min_y = min(min_y, landmark_y)
                                max_x = max(max_x, landmark_x)
                                max_y = max(max_y, landmark_y)

                    # curr_landmarks_list = 모든 원소에서 body_center 좌표를 빼주기

                    second_count += 1
                    # start_time = time.time()
                    video_final_values = []  # 현재 비디오의 final_movement 값들을 저장할 리스트

                    # 직사각형의 대각선 길이 계산
                    rect_diagonal = sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)

                    # 움직임 계산
                    total_distance = 0
                    final_movement = 0  # 매 초마다 움직임을 저장할 변수 초기화
                    if prev_landmarks_list and curr_landmarks_list:
                        for prev, curr in zip(prev_landmarks_list, curr_landmarks_list):
                            total_distance += sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
                        if rect_diagonal > 0:  # 직사각형 대각선 길이가 0보다 클 때만 계산
                            final_movement = total_distance / (rect_diagonal * recognized_landmarks_count)
                        all_final_values.append(final_movement)
                        video_final_values.append(final_movement)


                    prev_landmarks_list = curr_landmarks_list

                    # cv2.imshow('MediaPipe Pose', image)
                    if cv2.waitKey(5) & 0xFF == 27:
                        break

                cap.release()
                
            if video_final_values:
                avg_final_value = sum(video_final_values) / len(video_final_values)
                
                # scores_df.append( [filename, avg_final_value] )
                
                # print(f"Average Final Value for {filename}: {avg_final_value:.4f}")
                
                # if filename in scores_df['video_file'].values:
                #     # movement_score 업데이트
                #     match = scores_df['video_file'] == filename                    
                # if match.any():
                #     scores_df.loc[match, 'movement_score'] = avg_final_value
                # else:
                #     print(f"Filename {filename} not found in CSV.")
                
                if filename not in scores_df['video_file'].values:
                    # 새로운 행 추가
                    # scores_df = scores_df.append({'video_file': filename, 'movement_score': avg_final_value}, ignore_index=True)
                    new_row = pd.DataFrame({'video_file': [filename], 'movement_score': [avg_final_value]})
                    scores_df = pd.concat([scores_df, new_row], ignore_index=True)

                    
                else:
                    # 'movement_score' 업데이트
                    scores_df.loc[scores_df['video_file'] == filename, 'movement_score'] = avg_final_value
                    
    
    # 변경 사항을 CSV 파일에 다시 저장
    scores_df.to_csv(output_csv_path, index=False)  
    print( f"End!" )
      
    cv2.destroyAllWindows()

write_movement_scores("test_video\\final_sleeping_splitted")
# write_movement_scores("test_video\\final_not_sleeping_splitted")