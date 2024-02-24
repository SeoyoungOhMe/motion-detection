import cv2
import mediapipe as mp
import numpy as np
import os
from math import sqrt
import time

# Mediapipe 솔루션 초기화
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# 비디오 파일이 있는 폴더 경로 설정
folder_path = "test_video\\0219notSleeping_combined"
all_final_values = []  # 모든 영상에서 계산된 final value들을 저장할 리스트


# 필요한 변수들을 초기화
center_points_sum_x = 0
center_points_sum_y = 0
center_points_count = 0

center_point_x_pixel = 0
center_point_y_pixel = 0

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.mp4', '.avi', '.mov')):
        video_path = os.path.join(folder_path, filename)
        cap = cv2.VideoCapture(video_path)

        # if not cap.isOpened():
        #     print(f"{filename}: EOF or Video Not Found.")
        #     continue

        print(f"\nProcessing {filename}...")
        
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

                if results.pose_landmarks:
                    for idx, landmark in enumerate(results.pose_landmarks.landmark):
                        
                        ### 코드 추가
                        # [11, 12, 23, 24] 인덱스에 해당하는 랜드마크의 좌표값 추출 및 중심점 계산
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
                            # 중심점을 이미지에 표시
                            # cv2.circle(image, (center_point_x_pixel, center_point_y_pixel), 10, (255, 0, 0), -1)


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

                second_count += 1
                # start_time = time.time()
                video_final_values = []  # 현재 비디오의 final_movement 값들을 저장할 리스트

                # # 직사각형의 대각선 길이 계산
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

                # 매 초마다 최소, 최대 좌표와 final movement 출력
                # print(f"{second_count} sec - Min X: {min_x}, Min Y: {min_y}, Max X: {max_x}, Max Y: {max_y}, Rect Diagonal: {rect_diagonal:.4f}, Final Movement: {final_movement:.4f}")
                # print(f"frame {second_count} - {final_movement:.4f}")

                prev_landmarks_list = curr_landmarks_list

                # cv2.imshow('MediaPipe Pose', image)
                # if cv2.waitKey(5) & 0xFF == 27:
                #     break

            cap.release()
            
        if video_final_values:
            avg_final_value = sum(video_final_values) / len(video_final_values)
            print(f"Average Final Value for {filename}: {avg_final_value:.4f}")
        else:
            print(f"No final movement data available for {filename}.")

cv2.destroyAllWindows()

if all_final_values:
    avg_final_value = sum(all_final_values) / len(all_final_values)
    print(f"\nAverage Final Value across all videos: {avg_final_value:.4f}")
else:
    print("\nNo final movement data available.")
