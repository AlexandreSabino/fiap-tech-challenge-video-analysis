import numpy as np

def is_anomaly(rgb_frame, cv2, face_mesh):
    img_rgb = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
    result_mesh = face_mesh.process(img_rgb)

    if result_mesh.multi_face_landmarks:
        for lm in result_mesh.multi_face_landmarks:
            if detect_anomaly_face(rgb_frame, lm.landmark):
                return True

    return False


def detect_anomaly_face(frame, landmarks):
    return detect_crooked_mouth(frame, landmarks)

def detect_crooked_mouth(frame, landmarks):
    """
    Detecta boca torta / careta baseada em simetria vertical dos lados da boca.
    """
    if not is_frontal(frame, landmarks):
        return False

    h, w, _ = frame.shape

    def p(idx):
        lm = landmarks[idx]
        return int(lm.x * w), int(lm.y * h)

    # Vertical esquerda (superior -> inferior)
    left_up = p(0)
    left_down = p(17)

    # Vertical direita (superior -> inferior)
    right_up = p(267)
    right_down = p(402)

    left_vertical = abs(left_down[1] - left_up[1])
    right_vertical = abs(right_down[1] - right_up[1])

    symmetry = abs(left_vertical - right_vertical)
    mouth_open = (left_vertical + right_vertical) / 2

    crooked = symmetry > mouth_open * 0.60

    return crooked

def is_frontal(frame, landmarks):

    # Pegando pontos dos olhos e nariz
    h, w, _ = frame.shape
    left_eye = landmarks[33]   # olho esquerdo
    right_eye = landmarks[263] # olho direito
    nose_tip = landmarks[1]    # ponta do nariz

    # Coordenadas normalizadas
    lx, ly = left_eye.x, left_eye.y
    rx, ry = right_eye.x, right_eye.y
    nx, ny = nose_tip.x, nose_tip.y

    # Diferença horizontal dos olhos
    eye_dx = abs(lx - rx)

    # Distância do nariz para o centro dos olhos
    eye_center_x = (lx + rx) / 2
    nose_offset = abs(nx - eye_center_x)

    frontal = (nose_offset / eye_dx) < 0.2
    return frontal
