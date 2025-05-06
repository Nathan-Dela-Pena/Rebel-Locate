# File: knn.py
import os
import numpy as np
import pandas as pd
from PIL import Image, ExifTags
from shapely.geometry import Point, Polygon, MultiPoint
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.dummy import DummyClassifier

CSV_PATH = "CS 422 - UNLV General Building Information - Sheet1.csv"
BOOTSTRAP_RADIUS = 0.00005

def extract_coordinates(image_path):
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        if not exif_data:
            return None
        gps_tag = next((tag for tag, name in ExifTags.TAGS.items() if name == "GPSInfo"), None)
        if gps_tag not in exif_data:
            return None
        gps_info = exif_data[gps_tag]

        def to_float(r):
            return float(r[0]) / float(r[1]) if isinstance(r, tuple) else float(r)

        def convert_to_degrees(value):
            d, m, s = value
            return to_float(d) + (to_float(m) / 60.0) + (to_float(s) / 3600.0)

        lat = convert_to_degrees(gps_info[2])
        lon = convert_to_degrees(gps_info[4])
        if gps_info[1] in ['S', 's']:
            lat = -lat
        if gps_info[3] in ['W', 'w']:
            lon = -lon

        return [lat, lon]
    except:
        return None

class CampusBuildingIdentifier:
    def __init__(self, k=3):
        self.k = k
        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.baseline = DummyClassifier(strategy='most_frequent')
        self.features = []
        self.labels = []
        self.coord_to_buildings, self.building_polygons = self.load_csv_geometry()

    def load_csv_geometry(self):
        df = pd.read_csv(CSV_PATH, skiprows=1)
        coord_map = {}
        poly_map = {}

        for _, row in df.iterrows():
            building_code = row.iloc[0]
            raw_coords = []

            for i, coord_str in enumerate(row[3:]):
                if i == 2:
                    continue
                if isinstance(coord_str, str) and ',' in coord_str:
                    try:
                        lat, lon = map(float, coord_str.strip().split(','))
                        pt = (round(lat, 6), round(lon, 6))
                        raw_coords.append(pt)
                        if pt not in coord_map:
                            coord_map[pt] = set()
                        coord_map[pt].add(building_code)
                    except:
                        continue

            if len(raw_coords) >= 3:
                polygon = MultiPoint(raw_coords).convex_hull
                poly_map[building_code] = polygon

        return coord_map, poly_map

    def determine_building(self, coords):
        rounded = (round(coords[0], 6), round(coords[1], 6))
        buildings = self.coord_to_buildings.get(rounded)
        if buildings:
            return next(iter(buildings))

        point = Point(coords)
        for building, polygon in self.building_polygons.items():
            if polygon.buffer(0.0001).contains(point):
                return building

        candidates = [
            b for coord, bs in self.coord_to_buildings.items()
            if abs(coord[0] - coords[0]) <= BOOTSTRAP_RADIUS and abs(coord[1] - coords[1]) <= BOOTSTRAP_RADIUS
            for b in bs
        ]
        if candidates:
            return sorted(candidates)[0]

        return None

    def load_feature_data(self, data_root):
        skipped = 0
        loaded = 0
        print(f"[INFO] Scanning {data_root} for images with GPS...")

        for root, _, files in os.walk(data_root):
            for fname in files:
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(root, fname)
                    coords = extract_coordinates(image_path)
                    if coords:
                        assigned = self.determine_building(coords)
                        if assigned:
                            self.features.append(coords)
                            self.labels.append(assigned)
                            loaded += 1
                            print(f"[LOAD] {image_path} → {assigned}")
                        else:
                            print(f"[SKIP] {image_path} → No building match")
                            skipped += 1
                    else:
                        print(f"[SKIP] {image_path} → No GPS")
                        skipped += 1

        print(f"[INFO] Loaded {loaded} samples with coordinates.")
        print(f"[WARN] Skipped {skipped} images.")

        if not self.features:
            print("[ERROR] No valid training data was found.")
        else:
            self.features, self.labels = resample(self.features, self.labels, random_state=42)

    def train_and_evaluate_kfold(self):
        if not self.features:
            print("[ERROR] No training data loaded.")
            return

        X = np.array(self.features)
        y = np.array(self.labels)

        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        accuracies = []

        for fold, (train_index, test_index) in enumerate(kf.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            self.knn.fit(X_train, y_train)
            preds = self.knn.predict(X_test)
            acc = accuracy_score(y_test, preds)
            accuracies.append(acc)
            print(f"Fold {fold + 1} Accuracy: {acc:.4f}")

        print(f"\nAverage KNN Accuracy over 10 folds: {np.mean(accuracies):.4f}")

        self.knn.fit(X, y)
        self.baseline.fit(X, y)

    def predict_building(self, feature_vector):
        prediction = self.knn.predict([feature_vector])[0]
        matched_building = self.determine_building(feature_vector)
        return prediction or matched_building
