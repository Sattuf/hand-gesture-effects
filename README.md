# ✋ Hand Gesture Effects

Real-time hand gesture recognition from your webcam. Each gesture triggers a different visual effect on the live video, built with **OpenCV** and **MediaPipe Hands**.

## Gestures and effects

| Gesture | Effect |
|---|---|
| ✊ Fist | Grayscale |
| ✌️ Peace sign | Blur |
| 👍 Thumbs up | Edge detection |
| 👌 OK sign | Inverted colours |
| 👆 Pointing | Cartoon |
| 🤟 Three fingers | Sepia |
| 🖖 Four fingers | Brightness boost |
| 🖐️ Open palm | Normal (no effect) |

## How it works

1. MediaPipe detects 21 hand landmarks in each frame.
2. Finger states are computed from landmark positions (handedness-aware for the thumb).
3. Special shapes (thumbs up, OK sign) are checked first, then the finger count is mapped to a gesture.
4. The matching OpenCV effect is applied to the frame, and the current gesture and effect are drawn on screen.

## Getting started

Requires Python 3.8–3.11 and a webcam.

```bash
pip install -r requirements.txt
python hand_gesture_effects.py
```

Press **Q** to quit.

## Tech stack

Python · OpenCV · MediaPipe · NumPy
