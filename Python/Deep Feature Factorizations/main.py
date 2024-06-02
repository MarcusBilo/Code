# pip install grad-cam
# pip install streamlit
# pip install "matplotlib<3.9"
import warnings
import numpy as np
import requests
import torch
from pytorch_grad_cam.utils.image import preprocess_image, show_factorization_on_image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import Imagenette
import streamlit as st
from pytorch_grad_cam import DeepFeatureFactorization, GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet18
from PIL import Image
from matplotlib.gridspec import GridSpec

# https://github.com/jacobgil/pytorch-grad-cam
# streamlit run C:/Users/Marcus/PycharmProjects/pythonProject1/main.py

st.set_page_config(layout="wide")

st.sidebar.header('Enter Numbers')
num1 = st.sidebar.number_input('index', min_value=0, step=1, value=0, format='%d')

num2 = st.sidebar.number_input('n comp', min_value=1, step=1, value=1, format='%d')
num3 = st.sidebar.number_input('top k', min_value=1, step=1, value=1, format='%d')

image_index = int(num1)
n_components = int(num2)
top_k = int(num3)


def get_image_from_file(image):
    img = np.array(image)
    rgb_img_float = np.float32(img) / 255
    input_tensor = preprocess_image(rgb_img_float,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    return img, rgb_img_float, input_tensor


def create_labels(concept_scores, top_k=2):
    imagenet_categories_url = \
        "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
    labels = eval(requests.get(imagenet_categories_url).text)
    concept_categories = np.argsort(concept_scores, axis=1)[:, ::-1][:, :top_k]
    concept_labels_topk = []
    for concept_index in range(concept_categories.shape[0]):
        categories = concept_categories[concept_index, :]
        concept_labels = [f"{labels[category].split(',')[0]}:{concept_scores[concept_index, category]:.2f}" for category
                          in categories]
        sorted_labels = sorted(concept_labels, key=lambda x: float(x.split(':')[1]), reverse=True)
        concept_labels_topk.append("\n".join(sorted_labels))
    return concept_labels_topk


plt.rcParams['figure.dpi'] = 90


def visualize_image(model, image, n_components=1, top_k=1):
    img, rgb_img_float, input_tensor = get_image_from_file(image)
    classifier = model.fc
    dff = DeepFeatureFactorization(model=model, target_layer=model.layer4,
                                   computation_on_concepts=classifier)
    concepts, batch_explanations, concept_outputs = dff(input_tensor, n_components)

    concept_outputs = torch.softmax(torch.from_numpy(concept_outputs), axis=-1).numpy()
    concept_label_strings = create_labels(concept_outputs, top_k=top_k)
    # Modify the list to add line breaks for two-word labels
    modified_concept_labels = [label.replace(" ", "\n") if " " in label else label for label in concept_label_strings]
    visualization = show_factorization_on_image(rgb_img_float,
                                                batch_explanations[0],
                                                image_weight=0.5,
                                                concept_labels=modified_concept_labels)

    result = np.hstack((img, visualization))
    return result


warnings.filterwarnings('ignore')
model = resnet18(pretrained=True)

transform = transforms.Compose([
    transforms.Resize((320, 320)),
])

train_data = Imagenette(root="Imagenette/train", split="train", size="320px", transform=transform)  # 9469
val_data = Imagenette(root="Imagenette/val", split="val", size="320px", transform=transform)  # 3925

label_map = {
    0: "tench",
    1: "English springer",
    2: "cassette player",
    3: "chain saw",
    4: "church",
    5: "French horn",
    6: "garbage truck",
    7: "gas pump",
    8: "golf ball",
    9: "parachute"
}

image, label = val_data[image_index]

model.eval()

# HiResCAM setup
rgb_img = np.float32(image) / 255
input_tensor = preprocess_image(rgb_img,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

input_tensor.requires_grad_()
outputs_hirescam = model(input_tensor)

target_layers = [model.layer4[-1]]
probabilities_hirescam = torch.nn.functional.softmax(outputs_hirescam, dim=1)
_, predicted_indices_hirescam = torch.topk(probabilities_hirescam, 3)
predicted_indices_hirescam = predicted_indices_hirescam.squeeze().tolist()
probabilities_np_hirescam = probabilities_hirescam.detach().numpy()
concept_labels_hirescam = create_labels(probabilities_np_hirescam, 1)

cam = HiResCAM(model=model, target_layers=target_layers)
grayscale_cam = cam(input_tensor=input_tensor)
grayscale_cam = grayscale_cam[0, :]
visualization_hirescam = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# DFF setup
_, _, input_tensor_dff = get_image_from_file(image)

for param in model.parameters():
    param.requires_grad = False

outputs_dff = model(input_tensor_dff)
probabilities_dff = torch.nn.functional.softmax(outputs_dff, dim=1)
_, predicted_indices_dff = torch.topk(probabilities_dff, 3)
predicted_indices_dff = predicted_indices_dff.squeeze().tolist()
concept_labels_dff = create_labels(probabilities_dff.numpy(), 3)

ima_dff = visualize_image(model, image, n_components=n_components, top_k=top_k)

# Create figure with GridSpec
fig = plt.figure(figsize=(12, 4))
gs = GridSpec(1, 2, width_ratios=[1, 3])  # 1:2 ratio

# HiResCAM image
ax1 = fig.add_subplot(gs[0])
ax1.imshow(visualization_hirescam)
ax1.axis('off')
ax1.set_title("HiResCAM")

# DFF image
ax2 = fig.add_subplot(gs[1])
ax2.imshow(ima_dff)
ax2.axis('off')
ax2.set_title("DFF")

# Adjust the spacing between subplots and margins around the plot
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()


# Streamlit display
if image is not None:
    st.pyplot(fig, use_container_width=True)

    # Create empty columns on either side of the main columns to center them
    col_left, col1, col2, col_right = st.columns([2, 2, 2, 1])

    with col1:
        st.write("Top predicted classes:")
        for i, labels in enumerate(concept_labels_hirescam[0].split("\n"), 1):
            st.write(f"{i}. {labels}")

    with col2:
        st.write("True class:\n")
        st.write(label_map[label])
