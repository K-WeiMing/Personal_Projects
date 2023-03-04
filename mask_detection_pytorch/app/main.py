# fastapi
from fastapi import FastAPI
from fastapi import UploadFile, File

# image processing / loading
from PIL import Image
from io import BytesIO

# uvicorn
import uvicorn

# torch
import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# typehinting
from typing import List, Dict


# initialize application
app = FastAPI(debug=True)


def transform_image(image_bytes: Image.Image):
    transform_img = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform_img(image_bytes)
    return img_tensor


def load_model(model_weights_path: str):
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)
    model.load_state_dict(torch.load(model_weights_path))
    model.eval()

    return model


def read_imagefile(file: bytes) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image


def process_prediction(pred: Dict[str, List[torch.Tensor]]) -> dict:
    """
    Processed the predicted output from the model

    Args:
        pred (Dict[str, List[torch.Tensor]]):
            Comes in the format of:
                {"boxes": [[...], [...], ...],
                "labels": [...],
                "scores": [...],
                }

    Returns:
        dict: {"boxes": ..., "labels": ..., "scores": ...} for scores > threshold
    """
    
    boxes, labels, scores = [], [], []

    for index, score in enumerate(pred["scores"]):
        score_np = float(score.detach().numpy())
        if score_np > 0.5:
            scores.append(score_np)
            labels.append(process_labels(pred["labels"][index].detach().numpy()))

            xmin, ymin, xmax, ymax = pred["boxes"][index]
            xmin = float(xmin.detach().numpy())
            ymin = float(ymin.detach().numpy())
            xmax = float(xmax.detach().numpy())
            ymax = float(ymax.detach().numpy())

            boxes.append([xmin, ymin, xmax, ymax])

    print(boxes, labels, scores)
    return {"boxes": boxes, "labels": labels, "scores": scores}


def process_labels(label: int) -> str:
    """
    Process labels and returns the classification

    Args:
        label (int): prediction from model

    Returns:
        str: classification of tagged label
    """

    if label == 1:
        return "with_mask"
    if label == 2:
        return "mask_weared_incorrect"


MODEL = load_model("model/mask_detection_fasterrcnn.pt")
MODEL.eval()


@app.get("/")
async def index():
    return {"Message": "This is index"}


@app.post("/predict/image")
async def predict(file: UploadFile = File(...)):
    img = await file.read()
    img = Image.open(BytesIO(img)).convert("RGB")
    img = transform_image(img)

    prediction = MODEL([img])

    # Process a single image output from MODEL
    processed_pred = process_prediction(prediction[0])

    return processed_pred

if __name__ == "__main__":
    uvicorn.run(app)
