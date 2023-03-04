## Aims of this mini-project:

1. To be able to apply transfer learning using Computer Vision on a publicly available dataset from Kaggle.

   - Aim to familiarize the building of a PyTorch Model using FasterRCNN (pre-trained)
   - Aim to successfully fine-tune the model
   - Aim to successfully load the saved model and perform inference

2. To be able to deploy the trained model using Docker/FastAPI
   - Aim to be familiar with using Docker and FastAPI
   - Deployment using both these tools for inference

## Running the application:

In your terminal, navigate to the folder `/app` and run: `uvicorn main:app` to launch the FastAPI endpoint.

Call the endpoint using the provided scripts (`send_img.py`) or through Postman.

## Dataset Source

https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
