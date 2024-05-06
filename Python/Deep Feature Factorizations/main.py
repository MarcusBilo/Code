import warnings

warnings.filterwarnings('ignore')
import numpy as np
import requests
import cv2
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
        concept_labels = []
        for category in categories:
            score = concept_scores[concept_index, category]
            label = f"{labels[category].split(',')[0]}:{score:.2f}"
            concept_labels.append(label)
        concept_labels_topk.append("\n".join(concept_labels))
    return concept_labels_topk


model = resnet50(pretrained=True)
model.eval()
print("Loaded model")


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

    # Just for the jupyter notebook, so the large images won't weight a lot:
    if result.shape[0] > 500:
        result = cv2.resize(result, (result.shape[1] // 4, result.shape[0] // 4))

    return result


image = (Image.fromarray(visualize_image(model,
                                        "https://th.bing.com/th/id/R.94b33a074b9ceeb27b1c7fba0f66db74?rik=wN27mvigyFlXGg&riu=http%3a%2f%2fimages5.fanpop.com%2fimage%2fphotos%2f31400000%2fBear-Wallpaper-bears-31446777-1600-1200.jpg&ehk=oD0JPpRVTZZ6yizZtGQtnsBGK2pAap2xv3sU3A4bIMc%3d&risl=&pid=ImgRaw&r=0",
                                        n_components=2)))

plt.imshow(image)
plt.show()
