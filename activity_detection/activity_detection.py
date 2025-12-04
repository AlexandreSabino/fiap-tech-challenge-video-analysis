import json

import cv2
import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from clip_creator import process_clips

VIDEO_PATH = "video_to_process.mp4"
OUTPUT_JSON = "activities.json"

MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"
processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
model = VideoMAEForVideoClassification.from_pretrained(MODEL_NAME)

print("Model loaded...")

def predict_clip(frames):
    frames_resized = [cv2.resize(f, (224, 224)) for f in frames]

    inputs = processor(
        images=frames_resized,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]
    probs = torch.softmax(logits, dim=0)

    # K =1
    values, indices = probs.topk(1)

    confidence = values[0].item()

    # Pelo menos 30% de confiança no movimento detectado
    if confidence > 0.30:
        return model.config.id2label[indices.item()]

all_actions = process_clips(video_path=VIDEO_PATH, detect_action=predict_clip)
with open(OUTPUT_JSON, "w") as f:
    json.dump(all_actions, f, indent=2)