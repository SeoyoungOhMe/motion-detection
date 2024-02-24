## 5개 랜드마크 

import os
import cv2
import mediapipe as mp

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

def detect_specific_landmarks(image, mp_pose):
    results = mp_pose.process(image)
    specific_landmarks = []

    if results.pose_landmarks:
        # 코, 왼쪽 손, 오른쪽 손, 왼쪽 발, 오른쪽 발 랜드마크 인덱스
        specific_landmarks_indices = [0, 7, 11, 22, 23]

        # 각 부위의 이름
        body_part_names = ["Nose", "Left Hand", "Right Hand", "Left Foot", "Right Foot"]

        for idx, landmark_idx in enumerate(specific_landmarks_indices):
            landmark = results.pose_landmarks.landmark[landmark_idx]
            h, w, c = image.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            specific_landmarks.append((body_part_names[idx], (x, y)))

            # 원 그리기
            cv2.circle(image, (x, y), 10, (0, 255, 0), -1)

            # 좌표와 부위 이름 출력
            cv2.putText(image, f'{body_part_names[idx]} : {x}, {y}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return specific_landmarks

if __name__ == "__main__":
    # 동영상 캡쳐
    video_path = "1sec-1.mp4"
    output_folder = "output_results-6"
    create_folder(output_folder)
    
    capture_frames(video_path, output_folder)

    # 이미지 로드
    image_0s = cv2.imread(os.path.join(output_folder, "frame_0s.jpg"))
    image_1s = cv2.imread(os.path.join(output_folder, "frame_1s.jpg"))

    # Mediapipe Pose 객체 생성
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Pose Detection
    landmarks_0s = detect_specific_landmarks(image_0s, pose)
    landmarks_1s = detect_specific_landmarks(image_1s, pose)

    # 랜드마크 좌표 출력
    print("Specific Landmarks 0s:")
    for body_part, coord in landmarks_0s:
        print(f"- {body_part} : {coord}")

    print("\nSpecific Landmarks 1s:")
    for body_part, coord in landmarks_1s:
        print(f"- {body_part} : {coord}")

    # 이미지 저장
    cv2.imwrite(os.path.join(output_folder, "frame_0s_specific_landmarks.jpg"), image_0s)
    cv2.imwrite(os.path.join(output_folder, "frame_1s_specific_landmarks.jpg"), image_1s)
