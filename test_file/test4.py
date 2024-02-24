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

        for idx in specific_landmarks_indices:
            landmark = results.pose_landmarks.landmark[idx]
            h, w, c = image.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            specific_landmarks.append((x, y))

    return specific_landmarks

def draw_specific_landmarks(image, landmarks):
    for landmark in landmarks:
        x, y = landmark
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

if __name__ == "__main__":
    # 동영상 캡쳐
    video_path = "1sec-2.mp4"
    output_folder = "output_results-2"
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
    print("Specific Landmarks 0s:", landmarks_0s)
    print("Specific Landmarks 1s:", landmarks_1s)

    # 이미지에 랜드마크 표시
    draw_specific_landmarks(image_0s, landmarks_0s)
    draw_specific_landmarks(image_1s, landmarks_1s)

    # 이미지 저장
    cv2.imwrite(os.path.join(output_folder, "frame_0s_specific_landmarks.jpg"), image_0s)
    cv2.imwrite(os.path.join(output_folder, "frame_1s_specific_landmarks.jpg"), image_1s)
