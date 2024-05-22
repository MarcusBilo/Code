# pip install grad-cam
# pip install streamlit
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

# https://github.com/jacobgil/pytorch-grad-cam
# streamlit run C:/Users/Marcus/PycharmProjects/pythonProject1/main.py


st.sidebar.header('Enter Numbers')
num1 = st.sidebar.number_input('index', min_value=0, step=1, value=0, format='%d')

# Model selection in the sidebar
model_choice = st.sidebar.selectbox("Choose a model", ["HiResCAM", "DFF"])

if model_choice == "DFF":
    num2 = st.sidebar.number_input('n comp', min_value=1, step=1, value=1, format='%d')
    num3 = st.sidebar.number_input('top k', min_value=1, step=1, value=1, format='%d')
else:
    num2 = 1
    num3 = 1

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

model.eval()

if model_choice == "DFF":

    _, _, input_tensor = get_image_from_file(image)

    for param in model.parameters():
        param.requires_grad = False

    outputs = model(input_tensor)

    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    _, predicted_indices = torch.topk(probabilities, 3)
    predicted_indices = predicted_indices.squeeze().tolist()
    concept_labels = create_labels(probabilities.numpy(), 3)

    f0 = plt.figure(0)
    ima = visualize_image(model, image, n_components=n_components, top_k=top_k)
    plt.imshow(ima)
    plt.show()

    if image is not None:
        st.pyplot(f0, use_container_width=True)
        col1, col2 = st.columns([2, 1])

        with col1:
            st.write("Top 3 predicted classes:")
            for i, labels in enumerate(concept_labels[0].split("\n"), 1):
                st.write(f"{i}. {labels}")

        with col2:
            st.write("True class:\n")
            st.write(label_map[label])
else:
    rgb_img = np.float32(image) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    input_tensor.requires_grad_()
    outputs = model(input_tensor)

    target_layers = [model.layer4[-1]]
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    _, predicted_indices = torch.topk(probabilities, 3)
    predicted_indices = predicted_indices.squeeze().tolist()
    probabilities_np = probabilities.detach().numpy()
    concept_labels = create_labels(probabilities_np, 1)

    cam = HiResCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    f1 = plt.figure(1)
    fig = Image.fromarray(visualization, 'RGB')
    plt.imshow(fig)
    plt.gcf().set_size_inches(4, 2)
    plt.show()

    if image is not None:
        st.pyplot(f1, use_container_width=False)
        col1, col2 = st.columns([2, 1])

        with col1:
            st.write("Top predicted class:")
            for i, labels in enumerate(concept_labels[0].split("\n"), 1):
                st.write(f"{i}. {labels}")

        with col2:
            st.write("True class:\n")
            st.write(label_map[label])
