import cv2
import dlib

def capture_frames(video_path, output_path_0s, output_path_1s):
    # 비디오 로드
    cap = cv2.VideoCapture(video_path)

    # 0초에서의 프레임 캡쳐
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame_0s = cap.read()
    cv2.imwrite(output_path_0s, frame_0s)

    # 1초 후의 프레임 캡쳐
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FPS)))
    ret, frame_1s = cap.read()
    cv2.imwrite(output_path_1s, frame_1s)

    # 비디오 캡쳐 객체 해제
    cap.release()
    

def detect_landmarks(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # 얼굴 탐지
    faces = detector(image, 1)

    if len(faces) > 0:
        face = faces[0]
        landmarks = predictor(image, face)

        # 코, 왼쪽 손, 오른쪽 손, 왼쪽 발, 오른쪽 발의 랜드마크 인덱스
        landmarks_indices = [30, 7, 9, 37, 46, 19, 24, 40, 43]

        # 랜드마크 좌표 저장
        landmarks_coordinates = [(landmarks.part(idx).x, landmarks.part(idx).y) for idx in landmarks_indices]

        return landmarks_coordinates
    else:
        return None

def draw_landmarks(image, landmarks):
    for point in landmarks:
        cv2.circle(image, point, 2, (0, 255, 0), -1)  # 점 그리기

if __name__ == "__main__":
    # 0초와 1초 때의 frame 캡쳐
    video_path = "1sec.mp4"
    output_path_0s = "frame_0s.jpg"
    output_path_1s = "frame_1s.jpg"

    capture_frames(video_path, output_path_0s, output_path_1s)

    # 이미지 로드
    image_0s = cv2.imread(output_path_0s)
    image_1s = cv2.imread(output_path_1s)

    # 랜드마크 탐지
    landmarks_0s = detect_landmarks(image_0s)
    landmarks_1s = detect_landmarks(image_1s)

    # 랜드마크가 있을 경우 좌표 출력 및 이미지에 표시
    if landmarks_0s and landmarks_1s:
        print("Landmarks 0s:", landmarks_0s)
        print("Landmarks 1s:", landmarks_1s)

        # 이미지에 랜드마크 표시
        draw_landmarks(image_0s, landmarks_0s)
        draw_landmarks(image_1s, landmarks_1s)

        # 이미지 저장
        cv2.imwrite("frame_0s_landmarks.jpg", image_0s)
        cv2.imwrite("frame_1s_landmarks.jpg", image_1s)










