## 매초마다 좌표 찍힌 프레임 저장하고 좌표값 출력

import cv2
import os
import time
import mediapipe as mp
from datetime import datetime

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

                # 원 그리기
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

                # 좌표와 랜드마크 이름 출력
                cv2.putText(image, f'{mp.solutions.pose.PoseLandmark(landmark_idx).name} : {x}, {y}', (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

    return specific_landmarks

def capture_and_show_frames(output_folder):
    # 카메라 열기
    cap = cv2.VideoCapture(0)  # 0은 기본 카메라를 나타냄

    # Mediapipe Pose 객체 생성
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    start_time = datetime.now().second
    before_time = start_time
    while True:
        # 현재 시간 계산
        current_time = datetime.now().second
        
        if (before_time == current_time) :
            continue

        # 프레임 읽기
        ret, frame = cap.read()

        # 화면에 프레임 표시
        cv2.imshow("Camera Feed", frame)

        # Mediapipe Pose를 사용하여 특정 위치의 랜드마크 표시
        specific_landmarks = detect_specific_landmarks(frame, pose, [0, 19, 20, 31, 32])

        # 1초마다 특정 작업 수행
        if int(current_time - start_time) % 1 == 0 :
            # 특정 랜드마크 좌표 출력
            specific_landmarks = detect_specific_landmarks(frame, pose, [0, 19, 20, 31, 32])
            print(f"\nSpecific Landmarks of {int(current_time - start_time)} sec:")
            for landmark, coord in specific_landmarks:
                print(f"- {landmark} : {coord}")

            # 이미지 저장
            frame_path = os.path.join(output_folder, f"frame_{int(current_time - start_time)}s.jpg")
            cv2.imwrite(frame_path, frame)

            before_time = current_time
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 카메라 닫기
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    output_folder = "frames-7"

    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    capture_and_show_frames(output_folder)
