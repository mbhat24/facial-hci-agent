# Architecture



Browser (getUserMedia)
│ WebSocket (JPEG frames @ 10 FPS)
▼
FastAPI server (Render)
│
├─► Perception:   MediaPipe FaceLandmarker (468 pts + 52 blendshapes)
├─► Features:     AU extraction, head pose, gaze, blink, temporal buffer
├─► Inference:    FACS rules → emotion + cognitive state
├─► Reasoning:    Groq Llama-3.3-70B → natural-language interpretation
└─► Personalize:  per-user baseline → adjusted thresholds
│
▼
WebSocket → Browser dashboard (live charts)

## Why browser-side camera?
Render servers have no webcam. The browser captures video, downsamples to
~10 FPS, compresses to JPEG, and ships frames over WebSocket. Server runs
MediaPipe on CPU (no GPU needed for Face Mesh).

## Performance targets
- End-to-end latency: < 200 ms per frame
- Frame rate: 8–12 FPS (enough for emotion; micro-expressions need 30 FPS)
- Memory: < 500 MB (fits Render free tier)
