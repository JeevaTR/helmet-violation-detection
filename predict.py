from ultralytics import YOLO

def run_prediction():
    model = YOLO("runs/detect/train/weights/best.pt")
    model.predict(
        source="test_images",
        conf=0.4,
        save=True
    )

if __name__ == "__main__":
    run_prediction()
