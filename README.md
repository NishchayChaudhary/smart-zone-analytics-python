
# Smart Zone Analytics

A real-time computer vision pipeline for detecting, tracking, and analyzing people inside predefined zones using YOLOv8, OpenCV, and Python.

## Overview

Smart Zone Analytics processes videos to detect people, assign persistent IDs using object tracking, and generate zone-based analytics such as entry counts, exit counts, and occupancy statistics. The project follows a modular architecture, making it easy to extend with additional analytics modules.

## Features

- YOLOv8 person detection
- Multi-object tracking (Centroid Tracker)
- Polygon-based zone definition
- Entry and exit counting
- Real-time occupancy tracking
- Annotated output video generation
- Streamlit dashboard
- YAML-based configuration
- Modular and extensible project structure

## Tech Stack

- Python 3
- Ultralytics YOLOv8
- OpenCV
- NumPy
- PyTorch
- Streamlit
- PyYAML

## Project Structure
smart-zone-analytics-python/
├── assets/
├── config/
├── data/
├── docs/
├── models/
├── notebooks/
├── outputs/
├── src/
│ ├── analytics/
│ ├── dashboard/
│ ├── detection/
│ ├── tracking/
│ └── utils/
├── tests/
├── videos/
├── main.py
├── requirements.txt
└── README.md


## Installation

Clone the repository:

```bash
git clone https://github.com/NishchayChaudhary/smart-zone-analytics-python.git
cd smart-zone-analytics-python
