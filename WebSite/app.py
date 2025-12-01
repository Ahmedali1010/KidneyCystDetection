import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template
from PIL import Image
from torchvision import transforms
import io

# --- 1. Define the EXACT same Model Class ---
class KidneyCystCNN(nn.Module):
    def __init__(self):
        super(KidneyCystCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1   = nn.Linear(256 * 8 * 8, 128) 
        self.fc2   = nn.Linear(128, 1) 
        self.relu  = nn.ReLU()
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = x.view(-1, 256 * 8 * 8) 
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x)) 
        return x

# --- 2. Load the Model and Define Transforms ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KidneyCystCNN().to(device)

# --- CRITICAL CHANGE HERE: Load the BEST model ---
try:
    model.load_state_dict(torch.load('kidney_cyst_model.pth'))
    print("✅ Success: Loaded 'kidney_cyst_model.pth'")
except FileNotFoundError:
    print("❌ Error: Could not find 'best_kidney_model.pth'.")
    print("   Make sure you copied the file to this folder!")

model.eval() # Set model to evaluation mode

# Define the same transforms as in training
data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Define your class names (must match your training)
class_names = ['Cyst', 'Non_Cyst'] 

# --- 3. Create the Flask App ---
app = Flask(__name__)

# This function applies transforms and gets a prediction
def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_tensor = data_transforms(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probability = output.item() # This is the prob of being class 1

    # Logic:
    # 0.0 - 0.5 = Class 0 (Cyst)
    # 0.51 - 1.0 = Class 1 (Non_Cyst)
    
    if probability > 0.5:
        prediction_class = class_names[1] # 'Non_Cyst'
        confidence = probability
    else:
        prediction_class = class_names[0] # 'Cyst'
        confidence = 1 - probability
        
    return prediction_class, confidence

# --- 4. Define the Routes ---

# This route serves your index.html file
@app.route('/')
def home():
    return render_template('index.html')

# This route handles the prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    image_bytes = file.read()
    
    try:
        prediction, confidence = predict_image(image_bytes)
        return jsonify({
            'prediction': prediction,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- 5. Run the App ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)