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

def detect_pose(image, mp_pose):
    results = mp_pose.process(image)
    landmarks = []

    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            h, w, c = image.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            landmarks.append((x, y))
            
    return landmarks

def draw_landmarks(image, landmarks):
    for landmark in landmarks:
        x, y = landmark
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

if __name__ == "__main__":
    # 동영상 캡쳐
    video_path = "1sec-2.mp4"
    output_folder = "output_results-1"
    create_folder(output_folder)
    
    capture_frames(video_path, output_folder)

    # 이미지 로드
    image_0s = cv2.imread(os.path.join(output_folder, "frame_0s.jpg"))
    image_1s = cv2.imread(os.path.join(output_folder, "frame_1s.jpg"))

    # Mediapipe Pose 객체 생성
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # Pose Detection
    landmarks_0s = detect_pose(image_0s, pose)
    landmarks_1s = detect_pose(image_1s, pose)

    # 랜드마크 좌표 출력
    print("Landmarks 0s:", landmarks_0s)
    print("Landmarks 1s:", landmarks_1s)

    # 이미지에 랜드마크 표시
    draw_landmarks(image_0s, landmarks_0s)
    draw_landmarks(image_1s, landmarks_1s)

    # 이미지 저장
    cv2.imwrite(os.path.join(output_folder, "frame_0s_landmarks.jpg"), image_0s)
    cv2.imwrite(os.path.join(output_folder, "frame_1s_landmarks.jpg"), image_1s)
