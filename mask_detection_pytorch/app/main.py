from fastapi import FastAPI
from fastapi import UploadFile, File
from pydantic import BaseModel
import uvicorn

import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

app = FastAPI()


def transform_image(image_bytes):
    transform_img = transforms.Compose([transforms.ToTensor()])


def plot_image(img_tensor, annotation):
    fig, ax = plt.subplots(1)
    img = img_tensor.cpu().data

    # Display the image
    ax.imshow(img.permute(1, 2, 0))

    for index, box in enumerate(annotation["boxes"]):
        xmin, ymin, xmax, ymax = box
        xmin = xmin.detach().numpy()
        ymin = ymin.detach().numpy()
        xmax = xmax.detach().numpy()
        ymax = ymax.detach().numpy()

        if annotation["labels"][index] == 1:
            color = "g"
        if annotation["labels"][index] == 2:
            color = "r"

        # Create a Rectangle patch
        if annotation["scores"][index].detach().numpy() > 0.5:
            rect = patches.Rectangle(
                (xmin, ymin),
                (xmax - xmin),
                (ymax - ymin),
                linewidth=2.5,
                edgecolor=color,
                facecolor="none",
            )

            # Add the patch to the Axes
            ax.add_patch(rect)
    ax.axis(False)
    plt.show()


def load_model(model_weights_path):
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()

    return model


MODEL = load_model("../model/mask_detection_fasterrcnn.pt")


@app.get("/")
async def index():
    return {"Message": "This is index"}


@app.post("/predict/image")
async def predict(file: UploadFile = File()):
    extension = file.filename.endswith(("jpg", "png"))

    if not extension:
        return "Image must be jpg or png format"
    img = read_imagefile(await file.read())
    prediction = predict(img)

    return prediction


if __name__ == "__main__":
    uvicorn.run(app)
