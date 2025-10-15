import torch
from torchvision import transforms
from PIL import Image
from models.cnn_model import SimpleCNN

def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleCNN()
    model.load_state_dict(torch.load("cnn_model.pth", map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        pred = output.argmax(dim=1, keepdim=True)

    print(f"🧠 Predicted class: {pred.item()}")

if __name__ == "__main__":
    predict("sample.png")
