// 각 초마다의 평균 움직임 점수를 계산하는 함수
function get_movement_score_for_file2(framesPerSecond: Frame[][]): number[] {
    // 각 초마다 계산된 평균 움직임 점수를 저장하는 배열
    let allSecondsFinalValues: number[] = [];
    
    // 각 초마다의 프레임들을 처리
    framesPerSecond.forEach(frames => {

        let allFinalValues: number[] = [];
        let prevLandmarksList: Frame = [];
        
        // 현재 초의 각 프레임을 처리
        frames.forEach((currFrame, frameIndex) => {
            // 인식된 랜드마크의 개수
            let recognizedLandmarksCount = 0;
            // 현재 프레임의 총 이동 거리
            let totalDistance = 0;
            // 최종 움직임 점수
            let finalMovement = 0;
            // 현재 프레임의 랜드마크들을 포함하는 직사각형의 최소/최대 x, y 좌표
            let min_x = Infinity, min_y = Infinity, max_x = 0, max_y = 0;
            
            // 현재 프레임의 각 랜드마크 처리
            currFrame.forEach((landmark, idx) => {
                if (prevLandmarksList.length > 0 && idx < prevLandmarksList.length) {
                    const prevLandmark = prevLandmarksList[idx];
                    // 이전 프레임과 현재 프레임의 랜드마크 사이의 거리 계산
                    totalDistance += Math.sqrt(Math.pow(landmark.x - prevLandmark.x, 2) + Math.pow(landmark.y - prevLandmark.y, 2));
                    
                    // 직사각형의 최소/최대 좌표 업데이트
                    min_x = Math.min(min_x, landmark.x);
                    min_y = Math.min(min_y, landmark.y);
                    max_x = Math.max(max_x, landmark.x);
                    max_y = Math.max(max_y, landmark.y);
                    
                    recognizedLandmarksCount++;
                }
            });
            
            // 직사각형의 대각선 길이 계산
            const rectDiagonal = Math.sqrt(Math.pow(max_x - min_x, 2) + Math.pow(max_y - min_y, 2));
            if (rectDiagonal > 0) {
                // 최종 움직임 점수 계산
                finalMovement = totalDistance / (rectDiagonal * recognizedLandmarksCount);
                allFinalValues.push(finalMovement);
            }
            
            // 현재 프레임을 이전 프레임으로 설정
            prevLandmarksList = currFrame;
        });
        
        // 현재 초의 움직임 점수들의 평균 계산
        if (allFinalValues.length > 0) {
            const avgFinalValue = allFinalValues.reduce((a, b) => a + b, 0) / allFinalValues.length;
            allSecondsFinalValues.push(avgFinalValue);
        } else {
            console.log("No final movement data available for this second.");
            allSecondsFinalValues.push(NaN);
        }
    });
    
    // 각 초마다의 평균 움직임 점수를 반환
    return allSecondsFinalValues;
}



// 가상의 랜드마크 데이터를 생성하는 함수
function createFakeLandmarkData(): Frame[][] {
    const framesPerSecond: Frame[][] = [];
    for (let second = 0; second < 3; second++) { // 3초간의 데이터 생성
        const frames: Frame[] = [];
        for (let frameNum = 0; frameNum < 30; frameNum++) { // 가정: 1초에 30프레임
            const frame: Frame = [];
            for (let landmarkNum = 0; landmarkNum < 32; landmarkNum++) { // 각 프레임에 32개의 랜드마크
                const landmark: Landmark = {
                    x: Math.random(), // 0과 1 사이의 임의의 값
                    y: Math.random(),
                    z: Math.random(),
                    visibility: Math.random()
                };
                frame.push(landmark);
            }
            frames.push(frame);
        }
        framesPerSecond.push(frames);
    }
    return framesPerSecond;
}

// 가상의 랜드마크 데이터를 생성
const fakeData = createFakeLandmarkData();

// 생성된 데이터를 사용하여 움직임 점수 계산
const movementScores = get_movement_score_for_file2(fakeData);

// 계산된 움직임 점수 출력
console.log("Movement Scores for each second:", movementScores);
