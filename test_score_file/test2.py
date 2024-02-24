## 내 1초 인식 코드로 SleepingBabies 넣어봄. => 0.0만 뜨고 s13.mp4에서 에러 뜸.

import os
import cv2
import mediapipe as mp
from math import sqrt

# 위에서 정의한 함수들(create_folder, capture_frames, detect_specific_landmarks, calculate_movement) 여기에 포함
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


# 폴더 내의 모든 동영상 파일에 대해 처리
def process_videos_in_folder(folder_path):
    video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi'))]
    final_values = []

    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        output_folder = os.path.join("frames", os.path.splitext(video_file)[0])
        create_folder(output_folder)

        # 프레임 캡쳐 및 에러 처리
        if not capture_frames(video_path, output_folder):
            print(f"Error capturing frames for video: {video_file}")
            continue  # 이 동영상은 건너뛰고 다음 동영상으로 넘어감

        # 이미지 로드 및 처리
        image_0s = cv2.imread(os.path.join(output_folder, "frame_0s.jpg"))
        image_1s = cv2.imread(os.path.join(output_folder, "frame_1s.jpg"))
        specific_landmarks_indices = [0, 19, 20, 31, 32]
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        specific_landmarks_0s = detect_specific_landmarks(image_0s, pose, specific_landmarks_indices)
        specific_landmarks_1s = detect_specific_landmarks(image_1s, pose, specific_landmarks_indices)
        movement = calculate_movement(specific_landmarks_0s, specific_landmarks_1s)
        h, w, _ = image_0s.shape
        frame_diagonal = sqrt(w ** 2 + h ** 2)
        final_value = movement / frame_diagonal
        final_values.append(final_value)

        # 결과 저장
        cv2.imwrite(os.path.join(output_folder, "frame_0s_specific_landmarks.jpg"), image_0s)
        cv2.imwrite(os.path.join(output_folder, "frame_1s_specific_landmarks.jpg"), image_1s)

        print(f"{video_file}: {final_value}")

    # 평균 final value 계산 및 출력
    if final_values:
        average_final_value = sum(final_values) / len(final_values)
        print(f"\nAverage Final Value: {average_final_value}")
    else:
        print("No videos processed.")

if __name__ == "__main__":
    folder_path = "sleepingBabies"
    print("Sleeping Babies : ")
    process_videos_in_folder(folder_path)
