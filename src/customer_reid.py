import cv2
import numpy as np
import torch
import torch.nn.functional as F

from torchreid.utils import FeatureExtractor


class CustomerReID:

    def __init__(self):

        self.extractor = FeatureExtractor(
            model_name="osnet_x1_0",
            device="cpu"
        )

        print("OSNet Customer Re-ID Ready!")

    # Extract person features
    def extract_features(self, person_image):

        if person_image is None or person_image.size == 0:
            return None

        # OpenCV BGR to RGB
        person_image = cv2.cvtColor(
            person_image,
            cv2.COLOR_BGR2RGB
        )

        # Extract 512-dimensional features
        features = self.extractor([person_image])

        # Normalize features
        features = F.normalize(features, p=2, dim=1)

        return features[0].cpu().numpy()

    # Compare two persons
    def compare_features(self, feature1, feature2):

        if feature1 is None or feature2 is None:
            return 0.0

        feature1 = torch.tensor(feature1).float().unsqueeze(0)
        feature2 = torch.tensor(feature2).float().unsqueeze(0)

        similarity = F.cosine_similarity(
            feature1,
            feature2
        )

        return similarity.item()
    
