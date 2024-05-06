import warnings
import numpy as np
import requests
import torch
from pytorch_grad_cam import DeepFeatureFactorization
from pytorch_grad_cam.utils.image import preprocess_image, show_factorization_on_image
from torchvision.models import resnet50
from PIL import Image
import matplotlib.pyplot as plt


# pip install grad-cam


def get_image_from_url(url):
    """A function that gets a URL of an image, 
    and returns a numpy image and a preprocessed
    torch tensor ready to pass to the model """

    img = np.array(Image.open(requests.get(url, stream=True).raw))
    rgb_img_float = np.float32(img) / 255
    input_tensor = preprocess_image(rgb_img_float,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    return img, rgb_img_float, input_tensor


def create_labels(concept_scores, top_k=2):
    """ Create a list with the image-net category names of the top scoring categories"""
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
    """ Create a list with the image-net category names of the top scoring categories,
    along with their scores"""
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


def visualize_image(model, img_url, n_components=5, top_k=2):
    img, rgb_img_float, input_tensor = get_image_from_url(img_url)
    classifier = model.fc
    dff = DeepFeatureFactorization(model=model, target_layer=model.layer4,
                                   computation_on_concepts=classifier)
    concepts, batch_explanations, concept_outputs = dff(input_tensor, n_components)

    concept_outputs = torch.softmax(torch.from_numpy(concept_outputs), axis=-1).numpy()
    concept_label_strings = create_labels(concept_outputs, top_k=top_k)
    visualization = show_factorization_on_image(rgb_img_float,
                                                batch_explanations[0],
                                                image_weight=0.3,
                                                concept_labels=concept_label_strings)

    result = np.hstack((img, visualization))
    return result


# ----------------------------------------------------------------------------------------------------------------------

warnings.filterwarnings('ignore')

model = resnet50(pretrained=True)
model.eval()

# ----------------------------------------------------------------------------------------------------------------------

_, _, input_tensor = get_image_from_url(
    "https://github.com/jacobgil/pytorch-grad-cam/blob/master/examples/both.png?raw=true")

with torch.no_grad():
    model.eval()
    outputs = model(input_tensor)

predicted_classes = create_labels_v2(outputs.numpy(), top_k=10)

print("Top predicted classes:")
for i, labels in enumerate(predicted_classes[0].split("\n"), 1):
    print(f"{i}. {labels}")

# ----------------------------------------------------------------------------------------------------------------------


f0 = plt.figure(0)

image = visualize_image(model,
                        "https://github.com/jacobgil/pytorch-grad-cam/blob/master/examples/both.png?raw=true",
                        n_components=2, top_k=2)
plt.imshow(image)
plt.show()
