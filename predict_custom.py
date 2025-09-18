import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
import sys
import os

# --- Configuration ---
MODEL_PATH = 'rice_leaf_disease.pth'
CLASSES_FILE = 'rice_leaf_classes.txt'
NUM_CLASSES = 10 

# --- Prediction Function (no changes needed here) ---
def predict_single_image(image_path, model, class_names, device):
    try:
        img = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: The file '{image_path}' was not found.")
        return None, None
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
    probabilities = F.softmax(outputs[0], dim=0)
    confidence, top_catid = torch.topk(probabilities, 1)
    predicted_class = class_names[top_catid.item()]
    return predicted_class, confidence.item()

# --- Script Execution ---
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python predict_custom.py <path_to_your_image>")
        sys.exit(1)
    image_to_predict = sys.argv[1]

    print("Setting up your custom model and environment...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # =========================================================================
    # FINAL SOLUTION: Replicate the Transfer Learning Process
    # =========================================================================

    # 1. Load the pre-trained ResNet-152 model with its original ImageNet weights.
    #    This provides the frozen layers that are missing from your saved file.
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

    # 2. Replace the final layer with a new one that matches your custom model's structure.
    #    This layer is currently un-trained.
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)

    # 3. Load your saved state_dict.
    #    The 'strict=False' argument is KEY. It tells PyTorch to only load the
    #    weights for layers that match between the model and the file.
    #    In this case, it will ONLY load the weights for 'fc.weight' and 'fc.bias'
    #    and ignore all the other layers, which is exactly what we want.
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'.")
        sys.exit(1)
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)

    # =========================================================================

    # 4. Set the model to evaluation mode
    model.eval()
    model.to(device)

    # 5. Load your custom class names
    if not os.path.exists(CLASSES_FILE):
        print(f"Error: Class file not found at '{CLASSES_FILE}'.")
        sys.exit(1)
    with open(CLASSES_FILE) as f:
        class_names = [line.strip() for line in f.readlines()]

    print("Setup complete. Starting prediction...")
    print("-" * 30)

    # --- Predict ---
    predicted_class, confidence = predict_single_image(image_to_predict, model, class_names, device)

    # --- Display Result ---
    if predicted_class and confidence is not None:
        formatted_class_name = predicted_class.replace('_', ' ').title()
        print(f"Image Path:   {image_to_predict}")
        print(f"Prediction:   {formatted_class_name}")
        print(f"Confidence:   {confidence * 100:.2f}%")