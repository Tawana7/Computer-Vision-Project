from ultralytics import YOLOWorld
import cv2, os

model = YOLOWorld("yolov8s-worldv2.pt")
test_prompts = ["straw", "drinking straw", "plastic straw"]

for prompt in test_prompts:
    model.set_classes(["plastic boba cup", "disposable coffee cup", prompt])
    n_straws = 0
    for f in os.listdir("review_batches/train_pool_assisted")[:15]:
        img = cv2.imread(os.path.join("review_batches/train_pool_assisted", f))
        results = model.predict(img, conf=0.05, verbose=False)[0]  # note: lower conf than before
        n_straws += sum(1 for c in results.boxes.cls if int(c) == 2)
    print(f"'{prompt}': {n_straws} straw detections across 15 images")