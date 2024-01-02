import json
import click
import torch
import torch.nn as nn
import os
import cv2
import shutil

from PIL import Image
from glob import glob
from torchvision import datasets, models, transforms
from pathlib import Path
from typing import Dict
from tqdm import tqdm


def detect(img_path: str) -> Dict[str, int]:
    images_list = []
    leaves_count = {"aspen": 0, "birch": 0, "hazel": 0, "maple": 0, "oak": 0}
    class_names = list(leaves_count.keys())

    # Extract singular leaves form image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 90)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)

        x -= 10
        y -= 10
        w += 20
        h += 20

        if cv2.contourArea(contour) > 225:
            object_image = img[y: y + h, x: x + w]

            images_list.append(object_image)

    # Read training dataset classes
    data_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load trained neural network model
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 5)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    loaded_data = torch.load("model.pth")
    model.load_state_dict(loaded_data)
    model.eval()

    # Process images with neural network
    for image in images_list:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        input_tensor = data_transform(pil_image)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to("cuda")

        with torch.no_grad():
            output = model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        _, top_catid = torch.topk(probabilities, 2)

        leaves_count[class_names[top_catid[0]]] += 1

    return leaves_count


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path), required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        leaves = detect(str(img_path))
        results[img_path.name] = leaves

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
