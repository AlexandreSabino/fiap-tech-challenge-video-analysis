import json
import os
import tempfile

import cv2
from fpdf import FPDF


class People:

    def __init__(self, id, first_emotion, is_anomaly, face_image):
        self.id = id
        self.emotions: set = set()
        self.emotions.add(first_emotion)
        self.last_emotion = first_emotion
        self.face_image = face_image
        if is_anomaly:
            self.anomaly_count = 1
        else:
            self.anomaly_count = 0

    def add_emotion(self, emotion, is_anomaly):
        self.emotions.add(emotion)
        if is_anomaly:
            self.anomaly_count += 1

        # Capturar mudanças bruscas no "humor"
        if self.last_emotion in ['sad', 'angry', 'surprise', 'disgust'] and emotion in ['happy']:
            self.anomaly_count += 1

        self.last_emotion = emotion

    def is_anomaly(self):
        return self.anomaly_count > 4


class Report:

    def __init__(self):
        self.all_peoples: dict = {}

    def set_emotion(self, id, emotion, is_anomaly, face_image):
        people = self.all_peoples.get(id)
        if people is None:
            self.all_peoples[id] = People(id, emotion, is_anomaly, face_image)
        else:
            self.all_peoples[id].add_emotion(emotion, is_anomaly)

    def count_people_by_emotions(self):
        count_people = {}
        for people in self.all_peoples.values():
            for emotion in people.emotions:
                count_people[emotion] = count_people.get(emotion, 0) + 1

        return count_people

    def count_people_with_anomaly(self):
        count = 0
        for people in self.all_peoples.values():
            if people.is_anomaly():
                count += 1
        return count

    def generate_report_pdf(self, total_frames, output_file="report.pdf"):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)

        pdf.cell(0, 10, "Relatório final", ln=True, align="C")
        pdf.ln(10)

        total_people = len(self.all_peoples)
        total_emotions = sum(len(p.emotions) for p in self.all_peoples.values())
        total_anomalies = self.count_people_with_anomaly()

        pdf.set_font("Arial", "", 12)

        pdf.cell(0, 10, f"Quantidade de pessoas: {total_people}", ln=True)
        pdf.cell(0, 10, f"Total de emoções registradas: {total_emotions}", ln=True)
        pdf.cell(0, 10, f"Total de anomalias detectadas: {total_anomalies}", ln=True)
        pdf.cell(0, 10, f"Total de frames: {total_frames}", ln=True)
        pdf.ln(10)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Distribuição de emoções:", ln=True)
        pdf.set_font("Arial", "", 12)
        emotions_count = self.count_people_by_emotions()
        for emotion, count in emotions_count.items():
            pdf.multi_cell(0, 8, f"{emotion}: {count} pessoa(s)")
        pdf.ln(10)

        for people in self.all_peoples.values():
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, f"Pessoa ID: {people.id}", ln=True)

            if people.is_anomaly:
                face_image = people.face_image
            else:
                face_image = None

            if face_image is not None:
                tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                cv2.imwrite(tmp_file.name, face_image)
                pdf.image(tmp_file.name, w=50)
                pdf.ln(5)

            pdf.set_font("Arial", "", 12)
            emotions_str = ", ".join(people.emotions)
            pdf.multi_cell(0, 8, f"Emoções detectadas: {emotions_str}")
            pdf.ln(5)

        append_activities_to_pdf(pdf)
        pdf.output(output_file)


def append_activities_to_pdf(pdf, json_path="../activity_detection/activities.json"):
    with open(json_path, "r") as f:
        activities = json.load(f)

    if not activities:
        return

    # Adiciona título da seção
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Atividades detectadas", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", "", 12)
    for act in activities:
        time_sec = round(act["time"], 2)
        action_name = act["actions"]
        pdf.multi_cell(0, 8, f"{time_sec}s - {action_name}")
