import cv2
import mediapipe as mp
import numpy as np
mp_pose = mp.solutions.pose



### 수선의 발 산출 모듈
# A에서 BC에 내린 수선의 발 좌표 return
def find_perpendicular_foot(A, B, C): 
    
    # A,B,C => np.array([float, float])
    # return => np.array([float, float])
    
    BC = C - B
    BC_length_squared = np.dot(BC, BC)
    BA_dot_BC = np.dot(A - B, BC)
    t = BA_dot_BC / BC_length_squared
    foot = B + t * BC
    
    return foot



### 회전 변환 모듈
# point를 theta만큼 회전 변환, 이때는 원점 기준
def rotate_point(point, theta): 

    # point => np.array, theta => np.radians
    # return => np.array
    
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                 [np.sin(theta), np.cos(theta)]])
    rotated_point = np.dot(rotation_matrix, point)

    return rotated_point



### flip_detection 모듈 
def is_flipped(foot, nose, left_s, right_s): 

    # foot, nose, left_s, right_s => np.array([float, float])
    # return => bool

    # foot만큼 평행이동(foot이 원점으로)
    nose -= foot
    left_s -= foot
    right_s -= foot

    # nose와 y축 사이 각도 계산 => theta => np.radians
    angle = np.arctan2(nose[1], nose[0]) 
    if nose[1] < 0: 
        angle += 180
    theta = np.radians(angle)

    # left_s, right_s 회전 실행
    rotated_left_s = rotate_point(left_s, theta)
    rotated_right_s = rotate_point(right_s, theta)

    #print(f'rotated_left_s: {rotated_left_s}')
    #print(f'rotated_right_s: {rotated_right_s}')

    if rotated_left_s[0] < rotated_right_s[0]:
        return False
    else: 
        return True



### CAM_VER
def flip_detection_cam():
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # pose detection
            results = pose.process(image)
            if results.pose_landmarks:
                
                nose = np.array([results.pose_landmarks.landmark[0].x, results.pose_landmarks.landmark[0].y])
                left_s = np.array([results.pose_landmarks.landmark[11].x, results.pose_landmarks.landmark[11].y])
                right_s = np.array([results.pose_landmarks.landmark[12].x, results.pose_landmarks.landmark[12].y])
                
                # 수선의 발 좌표 구하기
                foot = find_perpendicular_foot(nose, left_s, right_s)
                
                # detect flip
                flip = is_flipped(foot, nose, left_s, right_s)
                if flip: 
                    print('Flipped!')
                    ##################################
                    # 추가적인 로직 작성
                    #
                    ##################################
                else:
                    print('NOT Flipped!')
                    ##################################
                    # 추가적인 로직 작성
                    #
                    ##################################
            else: 
                print('Cannot Detect')

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow('camera',image)
            
            if cv2.waitKey(5) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                break
    cap.release()
    cv2.destroyAllWindows()



### VIDEO VER.
def flip_detection_video(video_path):
    cap = cv2.VideoCapture(video_path)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                break
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # pose detection
            results = pose.process(image)
            if results.pose_landmarks:
                
                nose = np.array([results.pose_landmarks.landmark[0].x, results.pose_landmarks.landmark[0].y])
                left_s = np.array([results.pose_landmarks.landmark[11].x, results.pose_landmarks.landmark[11].y])
                right_s = np.array([results.pose_landmarks.landmark[12].x, results.pose_landmarks.landmark[12].y])
                
                # 수선의 발 좌표 구하기
                foot = find_perpendicular_foot(nose, left_s, right_s)
                
                # detect flip
                flip = is_flipped(foot, nose, left_s, right_s)
                if flip: 
                    print('Flipped!')
                    ##################################
                    # 추가적인 로직 작성
                    #
                    ##################################
                else:
                    print('NOT Flipped!')
                    ##################################
                    # 추가적인 로직 작성
                    #
                    ##################################
            else: 
                print('Cannot Detect')

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (480, 640))
            cv2.imshow('video',image)
             
            if cv2.waitKey(5) & 0xFF == 27:
                cap.release()
                cv2.destroyAllWindows()
                break
    cap.release()
    cv2.destroyAllWindows()


    