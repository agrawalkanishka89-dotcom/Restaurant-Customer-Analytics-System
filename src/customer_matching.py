from datetime import datetime, timedelta
from src.customer_reid import CustomerReID


class CustomerMatching:

    def __init__(self, session_minutes=30, similarity_threshold=0.75):

        self.reid = CustomerReID()

        self.session_minutes = session_minutes
        self.similarity_threshold = similarity_threshold

        self.customers = {}
        self.tracking_map = {}

        self.customer_counter = 0

        print("Customer Matching System Ready!")

    def find_matching_customer(self, features, event_time):

        best_customer_id = None
        best_similarity = 0

        for customer_id, data in self.customers.items():

            last_exit = data["last_exit"]

            if last_exit is None:
                continue

            time_difference = event_time - last_exit

            if time_difference < timedelta(0):
                continue

            if time_difference > timedelta(minutes=self.session_minutes):
                continue

            similarity = self.reid.compare_features(
                features,
                data["features"]
            )

            print(
                f"Returning Match: {customer_id} "
                f"Similarity: {similarity:.4f}"
            )

            if similarity > best_similarity:

                best_similarity = similarity
                best_customer_id = customer_id

        if best_similarity >= self.similarity_threshold:
            return best_customer_id

        return None

    def find_active_customer(self, features):

        best_customer_id = None
        best_similarity = 0

        for customer_id, data in self.customers.items():

            if data["last_exit"] is not None:
                continue

            similarity = self.reid.compare_features(
                features,
                data["features"]
            )
            print("DEBUG Similarity:", similarity)
            print(
                f"EXIT Match: {customer_id} "
                f"Similarity: {similarity:.4f}"
            )

            if similarity > best_similarity:

                best_similarity = similarity
                best_customer_id = customer_id

        print("Best EXIT Similarity:", best_similarity)

        if best_similarity >= self.similarity_threshold:
            return best_customer_id

        return None

    def assign_customer(self, tracking_id, person_image, event_time=None):

        if event_time is None:
            event_time = datetime.now()

        features = self.reid.extract_features(person_image)

        if features is None:
            return None

        customer_id = self.find_matching_customer(
            features,
            event_time
        )

        if customer_id is not None:

            self.customers[customer_id]["last_exit"] = None
            self.customers[customer_id]["features"] = features

            self.tracking_map[tracking_id] = customer_id

            print("Returning Customer:", customer_id)

            return customer_id

        self.customer_counter += 1

        customer_id = f"CUST{self.customer_counter:03d}"

        self.customers[customer_id] = {
            "features": features,
            "last_entry": event_time,
            "last_exit": None
        }

        self.tracking_map[tracking_id] = customer_id

        print("New Customer:", customer_id)

        return customer_id

    def mark_exit(self, tracking_id, person_image=None, event_time=None):

        if event_time is None:
            event_time = datetime.now()

        customer_id = self.tracking_map.get(tracking_id)

        if customer_id is None and person_image is not None:

            features = self.reid.extract_features(person_image)

            if features is not None:

                customer_id = self.find_active_customer(features)

                if customer_id is not None:

                    self.tracking_map[tracking_id] = customer_id

                    print("Customer matched at EXIT:", customer_id)

        if customer_id is None:
            print("WARNING: Customer could not be identified at EXIT.")
            return None

        self.customers[customer_id]["last_exit"] = event_time

        print("Customer Exit:", customer_id)

        return customer_id