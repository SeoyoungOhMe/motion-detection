type Landmark = {
    x: number;
    y: number;
    z: number;
    visibility: number;
  };
  
type Frame = Landmark[];

function get_movement_score_for_file( frames: Frame[]): number {
    let allFinalValues: number[] = [];
    let prevLandmarksList: Frame = [];
    
    frames.forEach((currFrame, frameIndex) => {
      let recognizedLandmarksCount = 0;
      let totalDistance = 0;
      let finalMovement = 0;
      let min_x = Infinity, min_y = Infinity, max_x = 0, max_y = 0;
      
      currFrame.forEach((landmark, idx) => {
        // Assuming landmarks filtering and center point calculation are handled externally
        
        if (prevLandmarksList.length > 0 && idx < prevLandmarksList.length) {
          const prevLandmark = prevLandmarksList[idx];
          totalDistance += Math.sqrt(Math.pow(landmark.x - prevLandmark.x, 2) + Math.pow(landmark.y - prevLandmark.y, 2));
          
          // Update min and max x, y for the rectangle calculation
          min_x = Math.min(min_x, landmark.x);
          min_y = Math.min(min_y, landmark.y);
          max_x = Math.max(max_x, landmark.x);
          max_y = Math.max(max_y, landmark.y);
          
          recognizedLandmarksCount++;
        }
      });
      
      // Calculate the diagonal of the rectangle encompassing all landmarks
      const rectDiagonal = Math.sqrt(Math.pow(max_x - min_x, 2) + Math.pow(max_y - min_y, 2));
      if (rectDiagonal > 0) {
        finalMovement = totalDistance / (rectDiagonal * recognizedLandmarksCount);
        allFinalValues.push(finalMovement);
      }
      
      prevLandmarksList = currFrame;
    });
    
    if (allFinalValues.length > 0) {
      const avgFinalValue = allFinalValues.reduce((a, b) => a + b, 0) / allFinalValues.length;
      return avgFinalValue;
    } else {
      console.log("No final movement data available.");
      return NaN;
    }
  }


// 예시 랜드마크 데이터 프레임
const poseFrames: Frame[] = [
    // 첫 번째 프레임
  [
    { x: 0.5, y: 0.3, z: 0.2, visibility: 0.99 }, // 랜드마크 1
    { x: 0.4, y: 0.2, z: 0.1, visibility: 0.98 }, // 랜드마크 2
    // ... 다른 랜드마크들
  ],
  // 두 번째 프레임
  [
    { x: 0.55, y: 0.35, z: 0.25, visibility: 0.97 }, // 랜드마크 1 (이동 발생)
    { x: 0.45, y: 0.25, z: 0.15, visibility: 0.96 }, // 랜드마크 2 (이동 발생)
    // ... 다른 랜드마크들
  ],
  // 세 번째 프레임
  [
    { x: 0.6, y: 0.4, z: 0.3, visibility: 0.95 }, // 랜드마크 1 (더 큰 이동 발생)
    { x: 0.5, y: 0.3, z: 0.2, visibility: 0.94 }, // 랜드마크 2 (이동 발생)
    // ... 다른 랜드마크들
  ],
  // 추가 프레임들...
];

// 함수 사용
const movementScore = get_movement_score_for_file(poseFrames);

console.log(`Calculated Movement Score: ${movementScore}`);