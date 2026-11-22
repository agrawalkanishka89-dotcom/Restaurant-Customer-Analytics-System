# Restaurant Customer Analytics System

An AI-powered Restaurant Customer Analytics System that uses Computer Vision and Deep Learning to detect customers, track their entry and exit, estimate gender, and analyze customer visits through an interactive dashboard.

## Features

- Real-time person detection using YOLO11.
- Customer tracking using ByteTrack.
- Person Re-Identification using OSNet.
- Returning customer identification.
- 30-minute customer session management.
- Gender classification using FairFace (UniFace).
- Automatic customer entry and exit detection.
- Customer records stored in CSV.
- Interactive analytics dashboard using Streamlit.
- Hourly and daily customer analysis.
- Peak-hour analysis.
- Gender analytics.
- Repeat customer and visit frequency analysis.
- Entry and exit history.
- CSV data download.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Camera and image processing |
| YOLO11 | Person detection |
| ByteTrack | Object tracking |
| OSNet / TorchReID | Person Re-Identification |
| UniFace / FairFace | Gender classification |
| PyTorch | Deep learning |
| NumPy | Numerical operations |
| Pandas | Data processing |
| Streamlit | Analytics dashboard |
| Plotly | Interactive charts |

## Project Structure

```text
Restaurant_managment_system/
│
├── config/
│   └── custom_bytetrack.yaml
│
├── dashboard/
│   └── app.py
│
├── data/
│   └── customer_records.csv
│
├── models/
│   └── yolo11n.pt
│
├── src/
│   ├── __init__.py
│   ├── customer_matching.py
│   ├── customer_reid.py
│   └── gender_classifier.py
│
├── tests/
│   ├── test_customer_matching.py
│   └── test_customer_reid.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Restaurant_managment_system
```

### 2. Create Virtual Environment

```bash
python -m venv restaurant_env
```

### 3. Activate Environment

Windows PowerShell:

```powershell
.\restaurant_env\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Install OSNet / TorchReID

Clone the official repository separately:

```bash
git clone https://github.com/KaiyangZhou/deep-person-reid.git
```

Navigate into the cloned repository and install it:

```bash
cd deep-person-reid
python -m pip install -e . --no-deps
```

The project also requires the OSNet model weights file `deep_person_reid.pth` in the configured location.

## How to Run

### Start Customer Detection System

Open PowerShell in the project directory:

```powershell
python main.py
```

The camera will start detecting and tracking customers.

### Start Analytics Dashboard

Open another PowerShell terminal:

```powershell
cd E:\Restaurant_managment_system
.\restaurant_env\Scripts\Activate.ps1
cd dashboard
python -m streamlit run app.py
```

## Customer Data

All customer entry and exit records are stored in:

```text
data/customer_records.csv
```

CSV columns:

- Customer_ID
- Tracking_ID
- Event
- Date
- Time
- Gender

The system starts with a fresh CSV file and adds new records during operation.

## Analytics Dashboard

The dashboard provides:

- Restaurant Overview
- Gender Analytics
- Hourly Customer Analysis
- Peak Hours
- Daily Customer Analysis
- Customer Visit Frequency
- Repeat Customer Analysis
- Customer Visit Details
- Entry/Exit History

## Important Notes

- A webcam is required for real-time detection.
- OSNet feature extraction currently runs on CPU.
- Gender classification provides estimated predictions and may not always be accurate.
- Customer records are excluded from Git tracking to protect customer data.
- YOLO and OSNet model weights must be available before running the application.

## Future Improvements

- Improve person re-identification accuracy.
- Add persistent customer identity storage.
- Improve analytics and reporting.
- Deploy the dashboard.
- Add database integration.

## Author

Developed as an AI and Data Science project.
