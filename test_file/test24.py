import os
import cv2
import mediapipe as mp
from math import sqrt
import time

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def detect_specific_landmarks(image, mp_pose, specific_landmarks_indices):
    results = mp_pose.process(image)
    specific_landmarks = []

    if results.pose_landmarks:
        # for landmark_idx in specific_landmarks_indices:
            # if results.pose_landmarks.landmark[landmark_idx].visibility > 0.0:
            #     landmark = results.pose_landmarks.landmark[landmark_idx]
            #     h, w, c = image.shape
            #     x, y = int(landmark.x * w), int(landmark.y * h)
            #     specific_landmarks.append((mp.solutions.pose.PoseLandmark(landmark_idx).name, (x, y)))

                landmarks_list = []
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    if landmark.visibility < 0.5:  # 감지하지 못한 값을 처리합니다.
                        continue
                    h, w, c = image.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    landmarks_list.append((cx, cy))
                    
                # 원 그리기
                cv2.circle(image, (cx, cy), 5, (0, 255, 0), -1)

                # 좌표와 랜드마크 이름 출력
                cv2.putText(image, f'{mp.solutions.pose.PoseLandmark(idx).name} : {cx}, {cy}', (cx, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)


                # # 원 그리기
                # cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

                # # 좌표와 랜드마크 이름 출력
                # cv2.putText(image, f'{mp.solutions.pose.PoseLandmark(landmark_idx).name} : {x}, {y}', (x, y - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

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
    video_path = "vod.mp4"
    output_folder = "frames-5"
    create_folder(output_folder)

    # Mediapipe Pose 객체 생성
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # 이전 프레임의 랜드마크 저장
    previous_specific_landmarks = None

    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    start_time = time.time()

    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            break

        # Pose Detection
        specific_landmarks_0s = detect_specific_landmarks(frame, pose, [0, 19, 20, 31, 32])

        # 현재 시간과 시작 시간 비교하여 1초마다의 결과 출력
        elapsed_time = time.time() - start_time
        if elapsed_time >= frame_number:
            print(f"\nFrame {frame_number}s:")
            for landmark, coord in specific_landmarks_0s:
                print(f"- {landmark} : {coord}")

            # 화면에 프레임 표시
            cv2.imshow("Camera Feed", frame)

            # 변화량 계산 및 출력
            if previous_specific_landmarks is not None:
                movement = calculate_movement(previous_specific_landmarks, specific_landmarks_0s)
                print(f"Total Movement: {movement}")

                # 프레임의 대각선 길이 계산
                h, w, _ = frame.shape
                diagonal_length = sqrt(w**2 + h**2)
                # print(f"Frame Diagonal Length: {diagonal_length}")

                # 최종 값 출력
                final_value = movement / diagonal_length
                print(f"Final Value: {final_value}")

            # 이전 프레임의 랜드마크 갱신
            previous_specific_landmarks = specific_landmarks_0s

            frame_number += 1

        # 'q' 키를 누르면 종료
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
