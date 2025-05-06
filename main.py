# File: main.py
import os
import torch
import shutil
from PIL import Image
from torchvision import transforms
from torchvision.models import alexnet, AlexNet_Weights

from knn import CampusBuildingIdentifier, extract_coordinates
from cnn import Places365Reduced, run_scene_classifier, load_scene_data, train_and_evaluate

# === Configuration ===
DATA_ROOT = "/Users/nathan/PycharmProjects/CS 422 Project/.venv"
TEST_IMAGE_DIR = os.path.join(DATA_ROOT, "test_images")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def classify_scene(image_path, model, device, class_names):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        predicted = torch.argmax(outputs, 1).item()
    return class_names[predicted]

def main():
    print("== Running KNN Building Identifier ==")

    knn_model = CampusBuildingIdentifier(k=3)
    knn_model.load_feature_data(DATA_ROOT)
    knn_model.train_and_evaluate_kfold()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n== Running Scene Classifier Based on KNN Prediction ==")
    for fname in os.listdir(TEST_IMAGE_DIR):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(TEST_IMAGE_DIR, fname)
        coords = extract_coordinates(image_path)
        if not coords:
            print(f"[SKIP] No GPS for {fname}")
            continue

        predicted_building = knn_model.predict_building(coords)

        print(f"\nImage: {fname}")
        print(f"Predicted Building: {predicted_building}")

        if predicted_building:
            building_path = os.path.join(DATA_ROOT, predicted_building)
            if os.path.isdir(building_path):
                copied_path = os.path.join(building_path, fname)
                if not os.path.exists(copied_path):
                    shutil.copy(image_path, copied_path)

                print(f"[INFO] Training CNN for building: {predicted_building}")

                image_paths, labels, class_names = load_scene_data(predicted_building)
                weights = AlexNet_Weights.IMAGENET1K_V1
                base_model = alexnet(weights=weights)
                for param in base_model.parameters():
                    param.requires_grad = False

                model = Places365Reduced(base_model, num_classes=len(class_names)).to(device)
                train_and_evaluate(model, image_paths, labels, class_names, transform, device)

                scene_label = classify_scene(copied_path, model, device, class_names)
                print(f"CNN Scene Label: {scene_label}")
            else:
                print(f"[SKIP] CNN folder not found for: {predicted_building}")
        else:
            print("[SKIP] No valid building prediction.")

if __name__ == "__main__":
    main()
