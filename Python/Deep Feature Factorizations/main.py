# pip install grad-cam
# pip install streamlit
import warnings
import numpy as np
import requests
import torch
from pytorch_grad_cam import DeepFeatureFactorization
from pytorch_grad_cam.utils.image import preprocess_image, show_factorization_on_image
from torchvision.models import resnet50, resnet18
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import Imagenette
import streamlit as st

# https://github.com/jacobgil/pytorch-grad-cam
# streamlit run C:/Users/Marcus/PycharmProjects/pythonProject1/main.py

"""

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


def create_labels_v2(concept_scores, top_k=5):
    imagenet_categories_url = \
        "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
    labels = eval(requests.get(imagenet_categories_url).text)
    concept_categories = np.argsort(concept_scores, axis=1)[:, ::-1][:, :top_k]
    concept_labels_topk = []
    for concept_index in range(concept_categories.shape[0]):
        categories = concept_categories[concept_index, :]
        concept_labels = [f"{labels[category].split(',')[0]}:{concept_scores[concept_index, category]:.2f}" for category
                          in categories]
        concept_labels_topk.append("\n".join(concept_labels))
    return concept_labels_topk


plt.rcParams['figure.dpi'] = 80


def visualize_image(model, image, n_components=1, top_k=1):
    img, rgb_img_float, input_tensor = get_image_from_file(image)
    classifier = model.fc
    dff = DeepFeatureFactorization(model=model, target_layer=model.layer4,
                                   computation_on_concepts=classifier)
    concepts, batch_explanations, concept_outputs = dff(input_tensor, n_components)

    concept_outputs = torch.softmax(torch.from_numpy(concept_outputs), axis=-1).numpy()
    concept_label_strings = create_labels(concept_outputs, top_k=top_k)
    visualization = show_factorization_on_image(rgb_img_float,
                                                batch_explanations[0],
                                                image_weight=0.5,
                                                concept_labels=concept_label_strings)

    result = np.hstack((img, visualization))
    return result


# ----------------------------------------------------------------------------------------------------------------------

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

image, label = train_data[image_index]
_, _, input_tensor = get_image_from_file(image)

model.eval()

for param in model.parameters():
    param.requires_grad = False

outputs = model(input_tensor)

predicted_classes = create_labels_v2(outputs.numpy(), top_k=3)

f0 = plt.figure(0)
ima = visualize_image(model, image, n_components=n_components, top_k=top_k)
plt.imshow(ima)
plt.show()

if image is not None:
    st.pyplot(f0, use_container_width=True)
    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("Top 3 predicted classes:")
        for i, labels in enumerate(predicted_classes[0].split("\n"), 1):
            st.write(f"{i}. {labels}")

    with col2:
        st.write("True class:\n")
        st.write(label_map[label])

"""

# ----------------------------------------------------------------------------------------------------------------------

# https://www.kaggle.com/code/antwerp/where-is-the-model-looking-for-gradcam-pytorch

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, HiResCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet18
from PIL import Image
import torch

warnings.filterwarnings('ignore')


def create_labels_v3(concept_scores, top_k=5):
    imagenet_categories_url = \
        "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
    labels = eval(requests.get(imagenet_categories_url).text)
    concept_categories = np.argsort(concept_scores, axis=1)[:, ::-1][:, :top_k]
    concept_labels_topk = []
    for concept_index in range(concept_categories.shape[0]):
        categories = concept_categories[concept_index, :]
        concept_labels = [f"{labels[category].split(',')[0]}:{concept_scores[concept_index, category]:.2f}" for category
                          in categories]
        concept_labels_topk.append("\n".join(concept_labels))
    return concept_labels_topk


# Load model resnet18
model = resnet18(pretrained=True)

# Pick up layers for visualization
target_layers = [model.layer4[-1]]

path = "ILSVRC2012_val_00009346.JPEG"
rgb_img = Image.open(path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((320, 320)),
])
rgb_img = transform(rgb_img)
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

#

# Forward pass the input through the model
with torch.no_grad():
    model.eval()
    outputs = model(input_tensor)

# Apply softmax to get probabilities
probabilities = torch.nn.functional.softmax(outputs, dim=1)

# Get the top predicted classes and their probabilities
top_k = 3
_, predicted_indices = torch.topk(probabilities, top_k)
predicted_indices = predicted_indices.squeeze().tolist()

# Get concept labels for the top predicted classes
concept_labels = create_labels_v3(probabilities.numpy(), top_k)

for i, labels in enumerate(concept_labels[0].split("\n"), 1):
    print(f"{i}. {labels}")

#

# cam = GradCAM(model=model, target_layers=target_layers)
cam = HiResCAM(model=model, target_layers=target_layers)

# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers) as cam:
#   ...

# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used.

grayscale_cam = cam(input_tensor=input_tensor)
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

fig = Image.fromarray(visualization, 'RGB')
plt.imshow(fig)
plt.show()
