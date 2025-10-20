import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List


class HandGestureRecognizer:
    """Main class for hand gesture recognition and effect application"""
    
    def __init__(self):
        # Initialize MediaPipe Hand Detection
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Configure hand detection
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            
            min_tracking_confidence=0.5
        )
        
        # Finger tip landmark indices
        self.FINGER_TIPS = {
            'thumb': 4,
            'index': 8,
            'middle': 12,
            'ring': 16,
            'pinky': 20
        }
        
        # Finger PIP (Proximal Interphalangeal) joint indices for comparison
        self.FINGER_PIPS = {
            'thumb': 2,
            'index': 6,
            'middle': 10,
            'ring': 14,
            'pinky': 18
        }
        
        self.current_gesture = "No Hand Detected"
        self.current_effect = "Normal"
        
    def count_fingers(self, hand_landmarks, handedness: str) -> int:
        """
        Count the number of extended fingers
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            handedness: "Left" or "Right" hand
            
        Returns:
            Number of extended fingers (0-5)
        """
        fingers_up = 0
        landmarks = hand_landmarks.landmark
        
        # Thumb detection (special case - uses x-coordinate)
        if handedness == "Right":
            # For right hand, thumb is up if tip is to the right of PIP
            if landmarks[self.FINGER_TIPS['thumb']].x > landmarks[self.FINGER_PIPS['thumb']].x:
                fingers_up += 1
        else:
            # For left hand, thumb is up if tip is to the left of PIP
            if landmarks[self.FINGER_TIPS['thumb']].x < landmarks[self.FINGER_PIPS['thumb']].x:
                fingers_up += 1
        
        # Other fingers detection (compare y-coordinates)
        for finger in ['index', 'middle', 'ring', 'pinky']:
            tip_y = landmarks[self.FINGER_TIPS[finger]].y
            pip_y = landmarks[self.FINGER_PIPS[finger]].y
            
            # Finger is up if tip is above PIP (lower y value)
            if tip_y < pip_y:
                fingers_up += 1
                
        return fingers_up
    
    def detect_thumbs_up(self, hand_landmarks) -> bool:
        """Detect thumbs up gesture"""
        landmarks = hand_landmarks.landmark
        
        # Thumb tip should be significantly above thumb base
        thumb_tip = landmarks[4]
        thumb_base = landmarks[2]
        wrist = landmarks[0]
        
        # Check if thumb is pointing up and other fingers are closed
        thumb_up = thumb_tip.y < wrist.y - 0.1
        
        # Check if other fingers are relatively closed
        other_fingers_closed = True
        for finger in ['index', 'middle', 'ring', 'pinky']:
            tip_y = landmarks[self.FINGER_TIPS[finger]].y
            pip_y = landmarks[self.FINGER_PIPS[finger]].y
            if tip_y < pip_y - 0.05:  # Finger significantly extended
                other_fingers_closed = False
                break
        
        return thumb_up and other_fingers_closed
    
    def detect_ok_sign(self, hand_landmarks) -> bool:
        """Detect OK sign (thumb and index finger forming circle)"""
        landmarks = hand_landmarks.landmark
        
        # Get thumb tip and index tip positions
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Calculate distance between thumb and index finger tips
        distance = np.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2
        )
        
        # Check if distance is small (forming circle) and other fingers are up
        fingers_touching = distance < 0.05
        
        # Middle, ring, and pinky should be extended
        middle_up = landmarks[12].y < landmarks[10].y
        ring_up = landmarks[16].y < landmarks[14].y
        pinky_up = landmarks[20].y < landmarks[18].y
        
        return fingers_touching and middle_up and ring_up
    
    def recognize_gesture(self, hand_landmarks, handedness: str) -> str:
        """
        Recognize the hand gesture based on finger positions
        
        Returns:
            String describing the recognized gesture
        """
        # Check for special gestures first
        if self.detect_thumbs_up(hand_landmarks):
            return "Thumbs Up"
        
        if self.detect_ok_sign(hand_landmarks):
            return "OK Sign"
        
        # Count extended fingers
        finger_count = self.count_fingers(hand_landmarks, handedness)
        
        # Map finger count to gestures
        gesture_map = {
            0: "Fist",
            1: "Pointing",
            2: "Peace Sign",
            3: "Three Fingers",
            4: "Four Fingers",
            5: "Open Palm"
        }
        
        return gesture_map.get(finger_count, "Unknown")
    
    def apply_effect(self, frame: np.ndarray, gesture: str) -> np.ndarray:
        """
        Apply image effect based on recognized gesture
        
        Args:
            frame: Input video frame
            gesture: Recognized gesture name
            
        Returns:
            Processed frame with applied effect
        """
        if gesture == "Fist":
            # Grayscale effect
            self.current_effect = "Grayscale"
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        elif gesture == "Peace Sign":
            # Blur effect
            self.current_effect = "Blur"
            return cv2.GaussianBlur(frame, (25, 25), 0)
        
        elif gesture == "Thumbs Up":
            # Edge detection
            self.current_effect = "Edge Detection"
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        elif gesture == "OK Sign":
            # Invert colors
            self.current_effect = "Inverted Colors"
            return cv2.bitwise_not(frame)
        
        elif gesture == "Pointing":
            # Cartoon effect
            self.current_effect = "Cartoon"
            return self.cartoonify(frame)
        
        elif gesture == "Three Fingers":
            # Sepia tone
            self.current_effect = "Sepia"
            return self.apply_sepia(frame)
        
        elif gesture == "Four Fingers":
            # Brightness boost
            self.current_effect = "Brightness Boost"
            return cv2.convertScaleAbs(frame, alpha=1.3, beta=50)
        
        elif gesture == "Open Palm":
            # Normal view
            self.current_effect = "Normal"
            return frame
        
        else:
            # No effect
            self.current_effect = "Normal"
            return frame
    
    def cartoonify(self, frame: np.ndarray) -> np.ndarray:
        """Apply cartoon effect to frame"""
        # Edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(
            gray_blur, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 9, 9
        )
        
        # Color quantization
        color = cv2.bilateralFilter(frame, 9, 300, 300)
        
        # Combine edges with color
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        return cartoon
    
    def apply_sepia(self, frame: np.ndarray) -> np.ndarray:
        """Apply sepia tone effect"""
        # Sepia transformation matrix
        sepia_matrix = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        
        sepia = cv2.transform(frame, sepia_matrix)
        sepia = np.clip(sepia, 0, 255)
        return sepia.astype(np.uint8)
    
    def draw_info(self, frame: np.ndarray, gesture: str, fps: float):
        """Draw information overlay on frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        
        # Draw text information
        cv2.putText(frame, f"Gesture: {gesture}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Effect: {self.current_effect}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw instructions at bottom
        instructions = "Press 'Q' to quit | Show different hand gestures to change effects"
        text_size = cv2.getTextSize(instructions, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = (width - text_size[0]) // 2
        cv2.putText(frame, instructions, (text_x, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Process a single frame: detect hands, recognize gesture, apply effect
        
        Returns:
            Tuple of (processed_frame, gesture_name)
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hands
        results = self.hands.process(rgb_frame)
        
        gesture = "No Hand Detected"
        
        if results.multi_hand_landmarks:
            # Process the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label
            
            # Recognize gesture
            gesture = self.recognize_gesture(hand_landmarks, handedness)
            
            # Apply effect before drawing landmarks
            frame = self.apply_effect(frame, gesture)
            
            # Draw hand landmarks on the processed frame
            self.mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
            )
        else:
            # No hand detected, show normal frame
            self.current_effect = "Normal"
        
        return frame, gesture
    
    def run(self):
        """Main loop: capture video, process frames, display results"""
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Hand Gesture Recognition System Started!")
        print("=" * 50)
        print("\nGesture Guide:")
        print("  🖐️  Open Palm (5 fingers) - Normal view")
        print("  ✊  Fist (0 fingers) - Grayscale")
        print("  ✌️  Peace Sign (2 fingers) - Blur")
        print("  👍  Thumbs Up - Edge Detection")
        print("  👌  OK Sign - Inverted Colors")
        print("  👆  Pointing (1 finger) - Cartoon Effect")
        print("  🤟  Three Fingers - Sepia Tone")
        print("  🖖  Four Fingers - Brightness Boost")
        print("\nPress 'Q' to quit")
        print("=" * 50)
        
        # FPS calculation variables
        prev_time = cv2.getTickCount()
        fps = 0
        
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            processed_frame, gesture = self.process_frame(frame)
            
            # Calculate FPS
            curr_time = cv2.getTickCount()
            time_diff = (curr_time - prev_time) / cv2.getTickFrequency()
            fps = 1.0 / time_diff if time_diff > 0 else 0
            prev_time = curr_time
            
            # Draw information overlay
            final_frame = self.draw_info(processed_frame, gesture, fps)
            
            # Display the frame
            cv2.imshow('Hand Gesture Recognition - Effects', final_frame)
            
            # Check for quit command
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print("\nProgram ended. Goodbye!")


def main():
    """Entry point of the program"""
    try:
        recognizer = HandGestureRecognizer()
        recognizer.run()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

