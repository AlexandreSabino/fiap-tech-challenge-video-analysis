import os

import mediapipe as mp
from deep_sort_realtime.deepsort_tracker import DeepSort
from deepface import DeepFace

from anomaly_detector import is_anomaly
from process_frame import process_video
from report import Report

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

tracker = DeepSort(max_age=30, n_init=1)
report = Report()


def detect_faces_and_emotions(rgb_frame, frame, cv2):
    # detector_backend = opencv Não detectou rostos laterais
    # detector_backend = mediapipe melhorou muito em relação ao opencv, porem ainda perde muitos frames.
    # detector_backend = retinaface parece muito melhor porem extremanente lento e usa muito recurso.
    try:
        results = DeepFace.analyze(
            rgb_frame,
            actions=['emotion'],
            detector_backend='yolov11n',
            enforce_detection=False,
            align=False,
            silent=True
        )

        detections = []

        for face in results:
            if face['face_confidence'] > 0.6:
                region = face['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']

                dominant_emotion = face['dominant_emotion']

                crop = rgb_frame[y:y + h, x:x + w]
                detections.append(([x, y, w, h], face['face_confidence'], crop))

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y + h), (x + w, y + h + 30), (0, 255, 0), -1)

                cv2.putText(
                    frame,
                    dominant_emotion,
                    (x + 5, y + h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

        identify_people(detections, results, rgb_frame, cv2)
    except Exception as e:
        print(f"Error in frame: {e}")


def identify_people(detections, results, rgb_frame, cv2):
    tracks = tracker.update_tracks(detections, frame=rgb_frame)
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = track.to_ltrb()

        matched_face = None
        for face in results:
            rx, ry, rw, rh = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            if abs(l - rx) < 20 and abs(t - ry) < 20:
                matched_face = face
                break

        if matched_face is None:
            continue

        region = matched_face['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        crop = rgb_frame[y:y + h, x:x + w]

        report.set_emotion(track_id,
                           matched_face['dominant_emotion'],
                           is_anomaly(rgb_frame, cv2, face_mesh),
                           crop)

def process_frame_to_frame(rgb_frame, frame, cv2):
    detect_faces_and_emotions(rgb_frame, frame, cv2)

script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'video_to_process.mp4')
output_video_path = os.path.join(script_dir, 'processed.mp4')

total_frames = process_video(video_path=input_video_path, output_path=output_video_path, process_frame=process_frame_to_frame)
report.generate_report_pdf(total_frames)
