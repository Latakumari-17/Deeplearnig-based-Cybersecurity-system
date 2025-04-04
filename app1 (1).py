import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import snntorch as snn
import torch.nn as nn
import numpy as np

# Define SNN Model
class SNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.lif1 = snn.Leaky(beta=0.9)
        self.fc2 = nn.Linear(128, 2)
        self.lif2 = snn.Leaky(beta=0.9)

    def forward(self, x, num_steps=25):
        mem1, mem2 = self.lif1.init_leaky(), self.lif2.init_leaky()
        x = x.view(x.size(0), -1)
        spk2_rec = []

        for _ in range(num_steps):
            spk1, mem1 = self.lif1(self.fc1(x), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk2_rec.append(spk2)

        return torch.stack(spk2_rec)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SNNModel().to(device)
model.load_state_dict(torch.load("malware_snn1.pth", map_location=device))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])
def file_to_binary_image(file, img_size=(64, 64)):
    """
    Converts a file into a binary (black & white) image.
    Each byte is mapped to either 0 (black) or 255 (white) based on its value.
    """
    content = file.read()
    byte_array = np.frombuffer(content, dtype=np.uint8)

    # Convert to binary: threshold at 128
    binary_array = np.where(byte_array > 127, 255, 0).astype(np.uint8)

    # Resize to fit the image
    binary_array = np.resize(binary_array, img_size[0] * img_size[1])
    image = binary_array.reshape(img_size)

    return image

# Streamlit App UI
st.title("Neuromorphic Malware Classifier 🧠🐛")
st.write("Upload an image representing malware or benign software for classification.")


uploaded_file =st.file_uploader("Upload any file", type=None)

img_size = st.slider("Image size (Width x Height)", 32, 256, 64)


if uploaded_file is not None:
    binary_img = file_to_binary_image(uploaded_file, img_size=(img_size, img_size))
    image = Image.fromarray(binary_img).convert("L")
    st.image(image, caption="🖼️ Generated Binary Image",use_container_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        spk_rec = model(img_tensor)
        output = spk_rec.sum(dim=0)
        _, predicted = output.max(1)

    label = "🛑 Malware Detected!" if predicted.item() == 1 else "✅ Benign (No Malware Found)"
    st.subheader("Classification Result:")
    if "Benign" in label:
        st.success(label)
    else:
        st.error(label)

