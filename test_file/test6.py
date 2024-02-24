## 33개 랜드마크(숫자로 출력)

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

def detect_all_landmarks(image, mp_pose):
    results = mp_pose.process(image)
    all_landmarks = []

    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            h, w, c = image.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            all_landmarks.append((f"Landmark {idx}", (x, y)))

            # 원 그리기
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

            # 좌표와 랜드마크 이름 출력
            cv2.putText(image, f'{idx} : {x}, {y}', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

    return all_landmarks

if __name__ == "__main__":
    # 동영상 캡쳐
    video_path = "1sec-2.mp4"
    output_folder = "output_results-7"
    create_folder(output_folder)
    
    capture_frames(video_path, output_folder)

    # 이미지 로드
    image_0s = cv2.imread(os.path.join(output_folder, "frame_0s.jpg"))
    image_1s = cv2.imread(os.path.join(output_folder, "frame_1s.jpg"))

    # Mediapipe Pose 객체 생성
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Pose Detection
    all_landmarks_0s = detect_all_landmarks(image_0s, pose)
    all_landmarks_1s = detect_all_landmarks(image_1s, pose)

    # 랜드마크 좌표 출력
    print("All Landmarks 0s:")
    for landmark, coord in all_landmarks_0s:
        print(f"- {landmark} : {coord}")

    print("\nAll Landmarks 1s:")
    for landmark, coord in all_landmarks_1s:
        print(f"- {landmark} : {coord}")

    # 이미지 저장
    cv2.imwrite(os.path.join(output_folder, "frame_0s_all_landmarks.jpg"), image_0s)
    cv2.imwrite(os.path.join(output_folder, "frame_1s_all_landmarks.jpg"), image_1s)
