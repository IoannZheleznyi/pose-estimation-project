import os
import time

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from sklearn.manifold import Isomap
import plotly.express as px

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# НАСТРОЙКИ
# =========================

VIDEOS_FOLDER = "/Users/ioann/KURSOVAYA/DS"
MODEL_PATH = "models/face_landmarker.task"

resize_factor = 0.5
frame_skip = 4


# =========================
# FACELANDMARKER
# =========================

options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)


# =========================
# DATAFRAME
# =========================

df = pd.DataFrame(
    columns=[
        "VideoNumber",
        "FrameNumber",
        "X",
        "Y",
        "Z",
        "Color"
    ]
)


video_files = sorted(
    [f for f in os.listdir(VIDEOS_FOLDER) if f.endswith(".mp4")]
)


# =========================
# ОБРАБОТКА ВИДЕО
# =========================

for video_file in video_files:

    video_path = os.path.join(VIDEOS_FOLDER, video_file)
    print(f"Processing: {video_path}")

    cap = cv2.VideoCapture(video_path)

    frame_number = 0

    video_df = pd.DataFrame(
        columns=[
            "X",
            "Y",
            "Z",
            "Color",
            "FrameNumber"
        ]
    )

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(
            frame,
            None,
            fx=resize_factor,
            fy=resize_factor
        )

        if frame_number % frame_skip != 0:
            frame_number += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect(mp_image)

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in landmarks]
            )

            center = coords.mean(axis=0)

            video_df.loc[len(video_df)] = [
                center[0],
                center[1],
                center[2],
                np.random.rand(),
                frame_number
            ]

        frame_number += 1

    cap.release()

    try:
        video_number = int(os.path.splitext(video_file)[0])
    except ValueError:
        continue

    video_df["VideoNumber"] = video_number

    df = pd.concat([df, video_df], ignore_index=True)


# =========================
# ЗАКРЫВАЕМ ДЕТЕКТОР
# =========================

detector.close()


# =========================
# ISOMAP
# =========================

if len(df) < 5:
    raise ValueError("Недостаточно данных для Isomap")

X_iso = Isomap(
    n_components=3,
    n_neighbors=min(5, len(df) - 1)
).fit_transform(df[["X", "Y", "Z"]])

df_iso = pd.DataFrame(
    X_iso,
    columns=["Dim1", "Dim2", "Dim3"]
)

df_iso["VideoNumber"] = df["VideoNumber"].values
df_iso["FrameNumber"] = df["FrameNumber"].values


# =========================
# 3D ГРАФИК (INTERACTIVE)
# =========================

fig = px.scatter_3d(
    df_iso,
    x="Dim1",
    y="Dim2",
    z="Dim3",
    color=df_iso["VideoNumber"].astype(str),
    hover_data={
        "VideoNumber": True,
        "FrameNumber": True,
        "Dim1": False,
        "Dim2": False,
        "Dim3": False
    },
    title="ISOMAP Projection of Face Centers"
)

fig.update_traces(marker=dict(size=4))

fig.show()