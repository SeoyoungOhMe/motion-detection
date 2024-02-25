function get_movement_score_for_file2( frames: Frame[]): number {

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
