import cv2
import time

from collections import deque, Counter
from uniface import FaceAnalyzer, FairFace


class GenderClassifier:

    def __init__(self):

        print("Loading FairFace Gender Model...")

        self.analyzer = FaceAnalyzer(
            predictors=[FairFace()]
        )

        # Store predictions for each tracking ID
        self.predictions = {}

        # Store last analysis time
        self.last_analysis = {}

        # Store final gender
        self.final_gender = {}

        # Analyze every 0.5 seconds
        self.analysis_interval = 0.5

        print("FairFace Gender Model Ready!")

    def predict_gender(self, tracking_id, person_crop):

        current_time = time.time()

        # Initialize tracking ID
        if tracking_id not in self.predictions:

            self.predictions[tracking_id] = deque(maxlen=7)
            self.last_analysis[tracking_id] = 0
            self.final_gender[tracking_id] = "Analyzing..."

        # Avoid repeated analysis
        if (
            current_time - self.last_analysis[tracking_id]
            < self.analysis_interval
        ):

            return self.final_gender[tracking_id]

        self.last_analysis[tracking_id] = current_time

        if person_crop is None or person_crop.size == 0:
            return "Unknown"

        try:

            faces = self.analyzer.analyze(person_crop)

            if not faces:
                return "Unknown"

            # Use the first detected face
            face = faces[0]

            gender = face.sex

            if gender is None:
                return "Unknown"

            gender = str(gender).strip().capitalize()

            if gender not in ["Male", "Female"]:
                return "Unknown"

            # Store prediction
            self.predictions[tracking_id].append(gender)

            # Majority voting after 7 predictions
            if len(self.predictions[tracking_id]) == 7:

                counts = Counter(
                    self.predictions[tracking_id]
                )

                self.final_gender[tracking_id] = (
                    counts.most_common(1)[0][0]
                )

            return self.final_gender[tracking_id]

        except Exception as e:

            print("Gender Prediction Error:", e)

            return "Unknown"
