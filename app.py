from flask import Flask, request, render_template
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
from torchvision.models import resnet18

app = Flask(__name__)

# Define transforms to apply to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 (input size for ResNet)
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# Create an instance of the ResNet model
model = resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 7)  # Assuming there are 7 classes
model.eval()

# Load the trained parameters
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))

# Get class labels
classes = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
           'cotton_bacterial_blight', 'cotton_curl_virus', 'cotton_fusarium_wilt', 'cotton_healthy']

# Define advisory messages for each class
advisories = {
    'Potato___Early_blight': "Apply fungicides like chlorothalonil or mancozeb at regular intervals. Improve drainage and avoid overhead watering.",
    'Potato___Late_blight': "Apply fungicides like copper oxychloride or zineb early and preventively. Destroy infected plant parts.",
    'Potato___healthy': "Your potato plants appear to be healthy. Continue to monitor for signs of disease and maintain good cultural practices.",
    'cotton_bacterial_blight': "Implement crop rotation. Apply copper-based bactericides like Bordeaux mixture or kasugamycin. Manage insect vectors like whiteflies..",
    'cotton_curl_virus': "Eliminate whitefly vectors with insecticides like imidacloprid or buprofezin. Use insect-resistant mesh and remove infected plants.",
    'cotton_fusarium_wilt': "Use resistant cotton varieties. Practice crop rotation and deep plowing. Avoid water stress and soil compaction.",
    'cotton_healthy': "Your cotton plants appear to be healthy. Continue to monitor for signs of disease and maintain good cultural practices."
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    img_file = request.files['image']

    # Read image from file
    img_bytes = img_file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    # Preprocess the image
    img = transform(img)

    # Add batch dimension
    img = img.unsqueeze(0)

    # Run inference
    with torch.no_grad():
        output = model(img)

    # Apply softmax to get probabilities
    probabilities = nn.functional.softmax(output[0], dim=0)

    # Get the index of the predicted class (disease) with the highest confidence score
    pred_index = torch.argmax(probabilities).item()

    # Get the predicted disease label and confidence score
    if pred_index < len(classes):
        pred_disease = classes[pred_index]
    else:
        pred_disease = "Unknown"

    # Round the confidence score to two decimal places
    confidence_score = round(probabilities[pred_index].item() * 100, 2)

    # Retrieve the advisory message for the predicted class
    advisory = advisories.get(pred_disease, "No specific advisory available.")

    # Encode the image as base64
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')

    # Pass prediction results and base64-encoded image data to results page
    return render_template('results.html', image=img_base64, disease=pred_disease, confidence=confidence_score,
                           advisory=advisory)


if __name__ == '__main__':
    app.run(debug=True)