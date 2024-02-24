## {영상 이름} : {정규화 값} 만 출력

import os
import cv2
import mediapipe as mp
from math import sqrt

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def capture_frames(video_path, output_folder):
    create_folder(output_folder)

    cap = cv2.VideoCapture(video_path)

    # 0초에서의 프레임 캡쳐
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame_0s = cap.read()
    cv2.imwrite(os.path.join(output_folder, "frame_0s.jpg"), frame_0s)

    # 1초 후의 프레임 캡쳐
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FPS)))
    ret, frame_1s = cap.read()
    cv2.imwrite(os.path.join(output_folder, "frame_1s.jpg"), frame_1s)

    cap.release()

def detect_specific_landmarks(image, mp_pose, specific_landmarks_indices):
    results = mp_pose.process(image)
    specific_landmarks = []

    if results.pose_landmarks:
        for landmark_idx in specific_landmarks_indices:
            if results.pose_landmarks.landmark[landmark_idx].visibility > 0.0:  # 좌표값이 구해진 경우에만 추가
                landmark = results.pose_landmarks.landmark[landmark_idx]
                h, w, c = image.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                specific_landmarks.append((mp.solutions.pose.PoseLandmark(landmark_idx).name, (x, y)))

                # print(w, h)
                                
                # 원 그리기
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

                # 좌표와 랜드마크 이름 출력
                cv2.putText(image, f'{mp.solutions.pose.PoseLandmark(landmark_idx).name} : {x}, {y}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

    return specific_landmarks

def calculate_movement(specific_landmarks_0s, specific_landmarks_1s):
    movement = 0.0
    valid_count = 0

    for (name_0s, (x_0s, y_0s)), (name_1s, (x_1s, y_1s)) in zip(specific_landmarks_0s, specific_landmarks_1s):
        # 구해진 좌표값이 있는 경우에만 계산에 포함
        if x_0s is not None and y_0s is not None and x_1s is not None and y_1s is not None:
            movement += sqrt((x_1s - x_0s) ** 2 + (y_1s - y_0s) ** 2)
            valid_count += 1

    # 랜드마크가 하나도 구해지지 않은 경우
    if valid_count == 0:
        return 0.0
    
    # 계산에 사용된 루트의 갯수만큼 전체 값을 나눠주도록 수정
    return round(movement / valid_count, 3)

    

if __name__ == "__main__":
    # 동영상 캡쳐
    video_path = "C:/motionDetection/test_video/1sec-2.mp4"
    output_folder = "frames-1"
    create_folder(output_folder)
    
    capture_frames(video_path, output_folder)

    # 이미지 로드
    image_0s = cv2.imread(os.path.join(output_folder, "frame_0s.jpg"))
    image_1s = cv2.imread(os.path.join(output_folder, "frame_1s.jpg"))

    # 지정한 랜드마크 인덱스
    specific_landmarks_indices = [0, 19, 20, 31, 32]

    # Mediapipe Pose 객체 생성
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Pose Detection
    specific_landmarks_0s = detect_specific_landmarks(image_0s, pose, specific_landmarks_indices)
    specific_landmarks_1s = detect_specific_landmarks(image_1s, pose, specific_landmarks_indices)

    # # 랜드마크 좌표 출력
    # print("Specific Landmarks 0s:")
    # for landmark, coord in specific_landmarks_0s:
    #     print(f"- {landmark} : {coord}")

    # print("\nSpecific Landmarks 1s:")
    # for landmark, coord in specific_landmarks_1s:
    #     print(f"- {landmark} : {coord}")

    # 이미지 저장
    cv2.imwrite(os.path.join(output_folder, "frame_0s_specific_landmarks.jpg"), image_0s)
    cv2.imwrite(os.path.join(output_folder, "frame_1s_specific_landmarks.jpg"), image_1s)

    # 변화량 계산 및 출력
    movement = calculate_movement(specific_landmarks_0s, specific_landmarks_1s)
    # print(f"\nTotal Movement: {movement}")

    # 이미지 차원(너비와 높이) 추출
    h, w, _ = image_0s.shape

    # 프레임의 대각선 길이 계산
    frame_diagonal = sqrt(w ** 2 + h ** 2)

    # 변화량을 대각선 길이로 나누어 최종 값 계산
    final_value = movement / frame_diagonal

    # 최종 값 출력
    print(f"{video_path}: {final_value}")
