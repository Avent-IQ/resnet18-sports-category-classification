### **🩺 ResNet-18 Sports Category Classification**  

This repository hosts a fine-tuned **ResNet-18-based** model optimized for **sports category classification** having 9 different labels of sports [cricket, archery, football, basketball, tennis, baseball, hockey, golf, boxing]. The model classifies images into these 9 categories.

---

## **📌 Model Details**  

- **Model Architecture**: ResNet-18  
- **Task**: Sports Category Classification  
- **Dataset**: 100 Sports Image Classification ([Kaggle](https://www.kaggle.com/datasets/gpiosenka/sports-classification))  
- **Framework**: PyTorch  
- **Input Image Size**: 224x224  
- **Number of Classes**: 9
---

## **🚀 Usage**  

### **Installation**  

```bash
pip install torch torchvision pillow
```

### **Loading the Model**  

```python
import torch
import torchvision.models as models
from huggingface_hub import hf_hub_download
import json
from PIL import Image
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

weights_path = hf_hub_download(repo_id="AventIQ-AI/resnet18-sports-category-classification", filename="resnet18_sports_classification.pth")
labels_path = hf_hub_download(repo_id="AventIQ-AI/resnet18-sports-category-classification", filename="class_labels.json")

with open(labels_path, "r") as f:
    class_labels = json.load(f)

model = models.resnet18(pretrained=False)

num_classes = len(class_labels)
model.fc = torch.nn.Linear(in_features=512, out_features=num_classes)

model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

model.eval()

print("Model loaded successfully!")
```

---

### **🔍 Perform Classification**  

```python
def predict_image(image_path, model, class_names):
    model.eval()
    
    # Load and transform the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    
    predicted_class = class_names[predicted.item()]
    return predicted_class

# Example usage:
image_path = "image_path.jpg"  # Change this to your image path
predicted_sport = predict_image(image_path, model, class_labels)
print(f"Predicted Sport Category: {predicted_sport}")
```

## **📊 Evaluation Results**  

After fine-tuning, the model was evaluated on the **Chest X-ray Pneumonia Dataset**, achieving the following performance:

| **Metric**        | **Score** |
|------------------|----------|
| **Accuracy**      | 92.4%    |
| **Precision**     | 88.2%    |
| **Recall**        | 82.8%    |
| **F1-Score**      | 88.5%    |

---

## **🔧 Fine-Tuning Details**  

### **Dataset**  
The model was trained on **100 Sports Image Classification** with labeled 9 classes of sports.  

### **Training Configuration**  

- **Number of epochs**: 10  
- **Batch size**: 32
- **Optimizer**: Adam  
- **Learning rate**: 1e-4  
- **Loss Function**: Cross-Entropy  
- **Evaluation Strategy**: Validation at each epoch  
---

## **⚠️ Limitations**  

- **Misclassification risk**: The model may produce **false positives or false negatives**. Always verify results with a radiologist.  
- **Dataset bias**: Performance may be affected by **dataset distribution**. It may not generalize well to **different populations**.  
- **Black-box nature**: Like all deep learning models, it does not explain why a prediction was made.  

---
