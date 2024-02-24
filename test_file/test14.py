import cv2
import os
import time

def capture_and_show_frames(output_folder):
    # 카메라 열기
    cap = cv2.VideoCapture(0)  # 0은 기본 카메라를 나타냄

    start_time = time.time()

    while True:
        # 현재 시간 계산
        current_time = time.time()

        # 프레임 읽기
        ret, frame = cap.read()

        # 화면에 프레임 표시
        cv2.imshow("Camera Feed", frame)

        # 프레임 파일 경로 생성
        frame_path = os.path.join(output_folder, f"{int(current_time - start_time)}s.jpg")

        # 프레임 저장
        cv2.imwrite(frame_path, frame)

        # 1초마다 메시지 출력
        if int(current_time - start_time) % 1 == 0:
            print(f"캡처된 프레임: {int(current_time - start_time)}초", end='\r', flush=True)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 카메라 닫기
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    output_folder = "frames-2"

    # 폴더가 존재하지 않으면 생성
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    capture_and_show_frames(output_folder)

    








