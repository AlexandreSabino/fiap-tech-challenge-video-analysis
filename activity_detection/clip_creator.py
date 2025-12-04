import cv2
from tqdm import tqdm

CLIP_LEN_FRAMES = 16

def get_all_video_frames_and_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames, fps

def chunk_frames(frames):
    for i in range(0, len(frames), CLIP_LEN_FRAMES):
        clip = frames[i:i + CLIP_LEN_FRAMES]

        if len(clip) < CLIP_LEN_FRAMES:
            clip += [clip[-1]] * (CLIP_LEN_FRAMES - len(clip))

        yield clip

def process_clips(video_path, detect_action):
    frames, fps = get_all_video_frames_and_fps(video_path)

    timestamp = 0.0
    actions = []
    num_clips = len(frames) // CLIP_LEN_FRAMES + 1

    for idx, clip in tqdm(enumerate(chunk_frames(frames)), total=num_clips):
        result = detect_action(clip)
        if result is not None:
            actions.append({
                "time": timestamp,
                "actions": result
            })
        timestamp = (idx * CLIP_LEN_FRAMES) / fps

    return actions