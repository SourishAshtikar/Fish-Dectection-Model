import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="train",
    label_types=["detections"],
    classes=["Fish"],
    max_samples=2000
)
