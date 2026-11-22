import cv2
import csv
import time
import shutil

from pathlib import Path
from datetime import datetime

from ultralytics import YOLO
from src.customer_matching import CustomerMatching
from src.gender_classifier import GenderClassifier


# ---------------------------------
# 1. Project Paths
# ---------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "yolo11n.pt"
CSV_PATH = BASE_DIR / "data" / "customer_records.csv"
TRACKER_PATH = BASE_DIR / "config" / "custom_bytetrack.yaml"

OLD_HEADERS = [
    "Tracking_ID",
    "Event",
    "Date",
    "Time"
]

PREVIOUS_HEADERS = [
    "Customer_ID",
    "Tracking_ID",
    "Event",
    "Date",
    "Time"
]

NEW_HEADERS = [
    "Customer_ID",
    "Tracking_ID",
    "Event",
    "Date",
    "Time",
    "Gender"
]


# ---------------------------------
# 2. CSV Setup and Migration
# ---------------------------------

if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:

    with open(CSV_PATH, "r", newline="") as file:
        reader = csv.DictReader(file)
        old_rows = list(reader)
        existing_headers = reader.fieldnames

    if existing_headers == NEW_HEADERS:

        print("CSV already has Gender column.")

    elif existing_headers in [OLD_HEADERS, PREVIOUS_HEADERS]:

        backup_path = BASE_DIR / "data" / "customer_records_before_gender.csv"

        if not backup_path.exists():
            shutil.copy2(CSV_PATH, backup_path)

        print("CSV backup created.")

        with open(CSV_PATH, "w", newline="") as file:

            writer = csv.DictWriter(
                file,
                fieldnames=NEW_HEADERS
            )

            writer.writeheader()

            for row in old_rows:

                writer.writerow({
                    "Customer_ID": row.get("Customer_ID", ""),
                    "Tracking_ID": row.get("Tracking_ID", ""),
                    "Event": row.get("Event", ""),
                    "Date": row.get("Date", ""),
                    "Time": row.get("Time", ""),
                    "Gender": "Unknown"
                })

        print("CSV successfully migrated.")

    else:

        raise SystemExit(
            "Unknown CSV format. Please check customer_records.csv."
        )

else:

    with open(CSV_PATH, "w", newline="") as file:

        writer = csv.writer(file)
        writer.writerow(NEW_HEADERS)


# ---------------------------------
# 3. Update Gender in Existing CSV
# ---------------------------------

def update_gender_records(tracking_id, gender):

    with open(CSV_PATH, "r", newline="") as file:

        reader = csv.DictReader(file)
        rows = list(reader)

    updated = False

    for row in rows:

        if str(row["Tracking_ID"]) == str(tracking_id):

            row["Gender"] = gender
            updated = True

    if updated:

        with open(CSV_PATH, "w", newline="") as file:

            writer = csv.DictWriter(
                file,
                fieldnames=NEW_HEADERS
            )

            writer.writeheader()
            writer.writerows(rows)


# ---------------------------------
# 4. Load YOLO Model
# ---------------------------------

model = YOLO(MODEL_PATH)

print("YOLO Model Loaded!")


# ---------------------------------
# 5. Initialize Customer Matching
# ---------------------------------

matcher = CustomerMatching(
    session_minutes=30,
    similarity_threshold=0.50
)


# ---------------------------------
# 6. Initialize Gender Classifier
# ---------------------------------

gender_classifier = GenderClassifier()

print("Gender Classification Ready!")


# ---------------------------------
# 7. Open Camera
# ---------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Camera could not be opened!")
    raise SystemExit

print("Camera Started!")


# ---------------------------------
# 8. Tracking Variables
# ---------------------------------

previous_sides = {}

candidate_sides = {}
candidate_counts = {}

last_event_times = {}

last_saved_gender = {}

total_entries = 0
total_exits = 0
current_occupancy = 0

REQUIRED_FRAMES = 3
COOLDOWN_SECONDS = 2

margin = 15


# ---------------------------------
# 9. Main Camera Loop
# ---------------------------------

while True:

    ret, frame = cap.read()

    if not ret:

        print("Camera frame not received.")
        break

    height, width = frame.shape[:2]

    line_y = int(height * 0.70)

    # YOLO + ByteTrack
    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=0.40,
        iou=0.50,
        tracker=str(TRACKER_PATH),
        verbose=False
    )

    current_time = time.time()

    for result in results:

        if result.boxes is None:
            continue

        if result.boxes.id is None:
            continue

        boxes = result.boxes

        for box in boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            tracking_id = int(box.id[0])

            box_width = x2 - x1
            box_height = y2 - y1

            # Ignore small detections
            if box_width < 40 or box_height < 70:
                continue

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # ---------------------------------
            # Create Person Crop
            # ---------------------------------

            crop_x1 = max(0, x1)
            crop_y1 = max(0, y1)

            crop_x2 = min(width, x2)
            crop_y2 = min(height, y2)

            person_crop = frame[
                crop_y1:crop_y2,
                crop_x1:crop_x2
            ]

            # ---------------------------------
            # Gender Prediction
            # ---------------------------------

            gender = gender_classifier.predict_gender(
                tracking_id,
                person_crop
            )

            # Update previous CSV records
            if gender in ["Male", "Female"]:

                if last_saved_gender.get(tracking_id) != gender:

                    update_gender_records(
                        tracking_id,
                        gender
                    )

                    last_saved_gender[tracking_id] = gender

            # ---------------------------------
            # Determine Person Side
            # ---------------------------------

            if center_y < line_y - margin:

                current_side = "TOP"

            elif center_y > line_y + margin:

                current_side = "BOTTOM"

            else:

                current_side = None

            # ---------------------------------
            # Initial Side Confirmation
            # ---------------------------------

            if tracking_id not in previous_sides:

                if current_side is not None:

                    if candidate_sides.get(tracking_id) == current_side:

                        candidate_counts[tracking_id] += 1

                    else:

                        candidate_sides[tracking_id] = current_side
                        candidate_counts[tracking_id] = 1

                    if candidate_counts[tracking_id] >= REQUIRED_FRAMES:

                        previous_sides[tracking_id] = current_side

                        candidate_sides.pop(tracking_id, None)
                        candidate_counts.pop(tracking_id, None)

            else:

                old_side = previous_sides[tracking_id]

                # ---------------------------------
                # Detect Side Change
                # ---------------------------------

                if current_side is not None and current_side != old_side:

                    if candidate_sides.get(tracking_id) == current_side:

                        candidate_counts[tracking_id] += 1

                    else:

                        candidate_sides[tracking_id] = current_side
                        candidate_counts[tracking_id] = 1

                    # Confirm crossing
                    if candidate_counts[tracking_id] >= REQUIRED_FRAMES:

                        event = None

                        if old_side == "TOP" and current_side == "BOTTOM":

                            event = "ENTRY"

                        elif old_side == "BOTTOM" and current_side == "TOP":

                            event = "EXIT"

                        previous_sides[tracking_id] = current_side

                        candidate_sides.pop(tracking_id, None)
                        candidate_counts.pop(tracking_id, None)

                        # ---------------------------------
                        # Cooldown Check
                        # ---------------------------------

                        last_event = last_event_times.get(
                            tracking_id,
                            0
                        )

                        if (
                            event is not None
                            and current_time - last_event >= COOLDOWN_SECONDS
                        ):

                            last_event_times[tracking_id] = current_time

                            event_time = datetime.now()

                            customer_id = None

                            # ---------------------------------
                            # ENTRY EVENT
                            # ---------------------------------

                            if event == "ENTRY":

                                customer_id = matcher.assign_customer(
                                    tracking_id,
                                    person_crop,
                                    event_time
                                )

                                total_entries += 1
                                current_occupancy += 1

                            # ---------------------------------
                            # EXIT EVENT
                            # ---------------------------------

                            elif event == "EXIT":

                                customer_id = matcher.mark_exit(
                                    tracking_id,
                                    person_crop,
                                    event_time
                                )

                                total_exits += 1

                                if current_occupancy > 0:

                                    current_occupancy -= 1

                            # ---------------------------------
                            # Save CSV Record
                            # ---------------------------------

                            with open(
                                CSV_PATH,
                                "a",
                                newline=""
                            ) as file:

                                writer = csv.writer(file)

                                writer.writerow([
                                    customer_id if customer_id else "",
                                    tracking_id,
                                    event,
                                    event_time.strftime("%Y-%m-%d"),
                                    event_time.strftime("%H:%M:%S"),
                                    gender
                                ])

                            print(
                                f"{event} | "
                                f"Customer: {customer_id} | "
                                f"Tracking ID: {tracking_id} | "
                                f"Gender: {gender}"
                            )

                else:

                    candidate_sides.pop(tracking_id, None)
                    candidate_counts.pop(tracking_id, None)

            # ---------------------------------
            # Draw Person Detection
            # ---------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"ID: {tracking_id}",
                (x1, y1 - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Gender: {gender}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

    # ---------------------------------
    # Draw Entry / Exit Line
    # ---------------------------------

    cv2.line(
        frame,
        (0, line_y),
        (width, line_y),
        (0, 0, 255),
        2
    )

    # ---------------------------------
    # Display Analytics
    # ---------------------------------

    cv2.putText(
        frame,
        f"Entries: {total_entries}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Exits: {total_exits}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"Occupancy: {current_occupancy}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.imshow(
        "Restaurant Customer Analytics",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ---------------------------------
# 10. Release Resources
# ---------------------------------

cap.release()
cv2.destroyAllWindows()


# ---------------------------------
# 11. Final Report
# ---------------------------------

print("\n========== FINAL REPORT ==========")

print("Total Entries:", total_entries)
print("Total Exits:", total_exits)
print("Current Occupancy:", current_occupancy)

print("Camera Closed Successfully!")