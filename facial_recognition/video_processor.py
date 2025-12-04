import os

from deepface import DeepFace

from process_frame import process_video


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

        for face in results:
            if face['face_confidence'] > 0.6:
                region = face['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                dominant_emotion = face['dominant_emotion']

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

    except Exception as e:
        print(f"Error in frame: {e}")


def process_frame_to_frame(rgb_frame, frame, cv2):
    detect_faces_and_emotions(rgb_frame, frame, cv2)


script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'video_to_process.mp4')
output_video_path = os.path.join(script_dir, 'processed.mp4')

process_video(video_path=input_video_path, output_path=output_video_path, process_frame=process_frame_to_frame)
