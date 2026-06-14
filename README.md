# Face Position Estimation and Visualization Using Computer Vision

## Overview

This project investigates computer vision techniques for estimating facial position in three-dimensional space from video streams. The primary objective is to develop an efficient pipeline for extracting facial coordinates, analyzing facial movement, and visualizing the resulting data.

The project combines facial landmark detection, video processing, dimensionality reduction, and interactive visualization to provide insights into facial motion patterns across video sequences.

---

## Why MediaPipe Instead of OpenPose?

Selecting an appropriate computer vision framework is a crucial step in developing a facial analysis system. For this project, **MediaPipe** was chosen over **OpenPose** due to several key advantages:

* Specialized functionality focused on facial landmark detection and tracking.
* Easy integration with Python applications.
* High processing speed and computational efficiency.
* Lightweight architecture suitable for real-time applications.
* Accurate facial landmark estimation.

As a result, MediaPipe provides an optimal balance between performance, simplicity, and accuracy for facial analysis tasks.

---

## Technologies Used

The project was developed in **Python** using the following libraries:

* **MediaPipe** – facial landmark detection and tracking.
* **OpenCV** – video capture and image processing.
* **NumPy** – numerical computations.
* **Pandas** – data storage and manipulation.
* **Scikit-learn** – dimensionality reduction using Isomap.
* **Matplotlib** – data visualization.
* **OpenPyXL** – Excel file export.

---

## Face Detection Pipeline

MediaPipe employs a computer vision and machine learning pipeline consisting of multiple interconnected models:

### 1. Face Detection Model

Detects and localizes the face within an image or video frame.

### 2. Facial Landmark Detection Model

Identifies key facial landmarks and reference points, enabling accurate estimation of facial position and orientation in three-dimensional space.

### 3. Face Tracking and Geometry Pipeline

Tracks facial landmarks across consecutive frames and provides stable coordinate measurements throughout the video stream.

Together, these components create an efficient pipeline capable of extracting detailed information about facial position, orientation, and movement.

---

## Dataset

The dataset consists of video recordings collected from TikTok. These videos contain faces captured under different conditions, expressions, poses, and viewing angles, providing diverse data for analysis.

---

## Data Collection Process

Each video is processed frame by frame using OpenCV and MediaPipe.

The workflow includes:

1. Loading video files from the dataset directory.
2. Extracting video frames.
3. Detecting facial landmarks.
4. Calculating representative facial coordinates (X, Y, Z).
5. Recording:

   * Video ID
   * Absolute frame number
   * Facial coordinates

The extracted information is stored in a Pandas DataFrame and exported to an Excel file for further analysis.

Output file:

```text
face_coordinates.xlsx
```

The resulting dataset can be used for visualization, machine learning experiments, and statistical analysis.

---

## Performance Optimization

To improve processing efficiency, several optimization parameters were introduced:

* **resize_factor** – scales video frames before processing.
* **frame_skip** – processes every fourth frame instead of every frame.
* **min_detection_confidence** – minimum confidence threshold required for face detection.

These parameters help balance computational performance and detection accuracy.

---

## Facial Landmark Visualization

Detected facial landmarks are rendered directly on processed video frames using MediaPipe visualization tools.

This enables real-time monitoring of facial position and movement throughout the video sequence.

---

## Dimensionality Reduction with Isomap

To simplify the analysis of three-dimensional facial coordinate data, the **Isomap** dimensionality reduction algorithm is applied.

The transformed data is represented in a reduced feature space consisting of:

* Dim1
* Dim2
* Dim3

This representation preserves important structural relationships while making the data easier to visualize and interpret.

---

## 3D Data Visualization

A three-dimensional scatter plot is generated using the Isomap-transformed data.

Visualization features include:

* Each point represents an individual video frame.
* Point colors correspond to different video files.
* Interactive hover information displays:

  * Video number
  * Frame number

This visualization provides an intuitive representation of facial motion patterns and relationships within the dataset.

---

## Results

The developed system successfully:

* Detects facial landmarks from video streams.
* Extracts three-dimensional facial coordinates.
* Stores structured facial data in Excel format.
* Reduces dimensionality using Isomap.
* Visualizes facial motion patterns in a 3D feature space.

The combination of MediaPipe, OpenCV, and Isomap demonstrates an efficient approach for facial position estimation and visual analysis.

---

## Installation

### Clone the Repository

```bash
git clone git@github.com:IoannZheleznyi/pose-estimation-project.git
cd pose-estimation-project
```

### Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Place video files in the dataset directory and run:

```bash
python main.py
```

The application will:

1. Load video files.
2. Detect facial landmarks.
3. Extract facial coordinates.
4. Save results to Excel.
5. Apply Isomap dimensionality reduction.
6. Generate a 3D visualization.

---

## Project Structure

```text
pose-estimation-project/
│
├── main.py
├── README.md
├── requirements.txt
├── face_coordinates.xlsx
│
├── models/
│   └── face_landmarker.task
│
├── videos/
│   └── *.mp4
│
└── results/
```

---

## Applications

This project can be applied in:

* Facial motion analysis
* Human-computer interaction
* Behavioral studies
* Emotion recognition research
* Machine learning dataset preparation
* Computer vision education and research

---

## Future Improvements

Potential future enhancements include:

* Real-time webcam processing.
* Facial expression classification.
* Emotion recognition systems.
* Deep learning-based feature extraction.
* Automatic gesture recognition.
* Large-scale dataset processing.
* Interactive web-based dashboards.
* Integration with neural network classifiers.

---

## Author

**Ioann Zheleznyi**

Student : majority - Artificial Intelligence
