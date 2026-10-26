# Classroom Pulse

Multimodal Attendance & Engagement Intelligence — Skill Wallet Capstone Project 5.

Offline **inference** Streamlit prototype with five dashboard tabs.

## Run locally (recommended for privacy)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Upload a classroom video (MP4/AVI/MOV), optionally upload a PCM WAV recording, and inspect attendance proxies, signal fusion, advisories, and exportable reports. Classroom images and webcam snapshots support face detection only; they cannot establish sustained attendance.

## Deploy on Streamlit Community Cloud

Push `app.py`, `engine.py`, and `requirements.txt` to a GitHub repository; choose `app.py` as entrypoint in Streamlit Community Cloud. **Cloud deployment is not 100% offline edge processing:** uploaded video/audio is sent to the hosted Streamlit server. For the strict privacy requirement, run on an offline classroom laptop and block outbound network access. Streamlit's webcam input is a snapshot, not continuous tracking.

## Technical limitations

- Tracks use centroid matching and OpenCV's bundled frontal-face cascade. They are anonymous, may switch IDs, and are **not verified student IDs**. Register labels P001 etc. are temporary track IDs.
- Presence uses sampled detected-frame duration divided by analyzed video duration; this can undercount occlusion and is not reliable for consequential attendance decisions.
- Audio categorization uses RMS, zero-crossing rate, and spectral-band energy. It **cannot** distinguish instructor lecture from student discussion, and does not implement trained VAD.
- Head pose, posture, and true attentiveness are **not measured**. The visual index is a face-detection-count proxy, and the fused score is **not validated engagement**.
- No external database, API calls, face recognition, or model downloads. Session state persists only while the Streamlit session lives. Uploaded video is held in a temporary file during analysis and deleted afterward.
- Video processing caps at 15,000 sampled frames for resource protection. Results are batch-processed, not live streaming.

## Responsible use

Obtain appropriate consent and comply with school privacy policies. Avoid punitive or individual monitoring. Use reports only for aggregate pedagogical reflection; confirm actual attendance by an approved manual or identity-aware method.
