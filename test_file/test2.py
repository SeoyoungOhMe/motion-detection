import cv2
import numpy as np
import tensorflow as tf

def load_pretrained_model():
    model = tf.saved_model.load("ssd_mobilenet_v2_coco/saved_model")
    return model

def detect_objects(image, model):
    input_tensor = tf.convert_to_tensor([image])
    detections = model(input_tensor)

    return detections

def draw_landmarks(image, landmarks):
    for landmark in landmarks:
        x, y = int(landmark[0]), int(landmark[1])
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

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

if __name__ == "__main__":
    # 동영상 캡쳐
    video_path = "1sec.mp4"
    output_path_0s = "frame_0s.jpg"
    output_path_1s = "frame_1s.jpg"
    capture_frames(video_path, output_path_0s, output_path_1s)

    # 이미지 로드
    image_0s = cv2.imread(output_path_0s)
    image_1s = cv2.imread(output_path_1s)

    # 객체 감지 모델 로드
    detection_model = load_pretrained_model()

    # 객체 감지
    detections_0s = detect_objects(image_0s, detection_model)
    detections_1s = detect_objects(image_1s, detection_model)

    # 코, 손, 발 랜드마크 좌표 추출
    landmarks_0s = []
    landmarks_1s = []

    for detection in detections_0s['detection_boxes'].numpy():
        x, y, w, h = detection
        landmarks_0s.append((x + w/2, y + h/2))

    for detection in detections_1s['detection_boxes'].numpy():
        x, y, w, h = detection
        landmarks_1s.append((x + w/2, y + h/2))

    # 랜드마크 좌표 출력
    print("Landmarks 0s:", landmarks_0s)
    print("Landmarks 1s:", landmarks_1s)

    # 이미지에 랜드마크 표시
    draw_landmarks(image_0s, landmarks_0s)
    draw_landmarks(image_1s, landmarks_1s)

    # 이미지 저장
    cv2.imwrite("frame_0s_landmarks.jpg", image_0s)
    cv2.imwrite("frame_1s_landmarks.jpg", image_1s)
