# Realtime CLI Inference and Study UI - Final Year Thesis

This repository contains the codebase for my Final Year Thesis project. It comprises a real-time machine learning backend for physiological stress regression and a web-based UI for conducting study protocols.

## Repository Structure

- `machine_learning/`: The core Python backend. Contains the data processing pipeline, model training scripts, and the real-time inference CLI server.
- `UI/`: The React-based frontend application used to conduct user study sessions.
- `data/`: Contains all raw datasets (videos, physiological signals, labels) and processed features. (Note: Most contents are ignored by `.gitignore` due to size).
- `Thesis/`: Contains the LaTeX source code and figures for the written thesis document.

## Getting Started

Please refer to the [QUICKSTART.md](./QUICKSTART.md) file for instructions on how to run the application, including:
1. Starting the backend server
2. Launching the study UI
3. Running the canonical model training pipelines

## Architecture Overview

The system is split into a **Backend** (`machine_learning`) and a **Frontend** (`UI`). 

The backend runs a fast inference server that receives video streams and outputs physiological stress regression scores using models trained on canonical physiological data. The frontend is a structured protocol runner that interfaces with the backend to collect data systematically.
