# pip install grad-cam
# pip install streamlit
# pip install "matplotlib<3.9"
import warnings
import numpy as np
import requests
import torch
from pytorch_grad_cam.utils.image import show_factorization_on_image, show_cam_on_image
import matplotlib.pyplot as plt
from torchvision.datasets import Imagenette
import streamlit as st
from pytorch_grad_cam import DeepFeatureFactorization, GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad, HiResCAM
from torchvision.models import resnet18
import os
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from tqdm import tqdm
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
from torch.utils.data.sampler import BatchSampler


# https://discuss.pytorch.org/t/load-the-same-number-of-data-per-class/65198/4
class BalancedBatchSampler(BatchSampler):
    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for _, ll_label in loader:
            self.labels_list.append(ll_label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {lti_label: np.where(self.labels.numpy() == lti_label)[0] for lti_label in self.labels_set}
        for sl in self.labels_set:
            np.random.shuffle(self.label_to_indices[sl])
        self.used_label_indices_count = {ul_label: 0 for ul_label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size


# https://discuss.pytorch.org/t/torch-utils-data-dataset-random-split/32209/4
class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def get_image_from_file(input_image):
    img = np.array(input_image)
    rgb_img_float = np.float32(img) / 255
    return img, rgb_img_float


def create_labels(concept_scores, no_of_categories=2):
    concept_categories = np.argsort(concept_scores, axis=1)[:, ::-1][:, :no_of_categories]
    concept_labels_topk = []
    for concept_index in range(concept_categories.shape[0]):
        categories = concept_categories[concept_index, :]
        concept_labels = [f"{get_label_from_numeric(label_map[category])}:{concept_scores[concept_index, category]:.2f}"
                          for category
                          in categories]
        sorted_labels = sorted(concept_labels, key=lambda x: float(x.split(':')[1]), reverse=True)
        concept_labels_topk.append("\n".join(sorted_labels))
    return concept_labels_topk


# https://jacobgil.github.io/pytorch-gradcam-book/Deep%20Feature%20Factorizations.html
def create_labels2(concept_scores, no_of_categories=2):
    imagenet_categories_url = \
        "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
    c_labels = eval(requests.get(imagenet_categories_url).text)
    concept_categories = np.argsort(concept_scores, axis=1)[:, ::-1][:, :no_of_categories]
    concept_labels_topk = []
    for concept_index in range(concept_categories.shape[0]):
        categories = concept_categories[concept_index, :]
        concept_labels = [f"{c_labels[category].split(',')[0]}:{concept_scores[concept_index, category]:.2f}" for category
                          in categories]
        sorted_labels = sorted(concept_labels, key=lambda x: float(x.split(':')[1]), reverse=True)
        concept_labels_topk.append("\n".join(sorted_labels))
    return concept_labels_topk


def get_label_from_numeric(numeric_label):
    imagenet_categories_url = \
        "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
    num_labels = eval(requests.get(imagenet_categories_url).text)
    num_label = num_labels.get(numeric_label, "Label not found")
    if ',' in num_label:
        num_label = num_label.split(',', 1)[0]  # Split at the first comma and take the part before it
    return num_label


def visualize_image(input_model, input_image, vis_input_tensor, no_of_components=1, no_of_categories=1):
    img, rgb_img_float = get_image_from_file(input_image)
    classifier = input_model.fc
    dff = DeepFeatureFactorization(model=input_model, target_layer=input_model.layer4, computation_on_concepts=classifier)
    concepts, batch_explanations, concept_outputs = dff(vis_input_tensor, no_of_components)
    concept_outputs = torch.softmax(torch.from_numpy(concept_outputs), dim=-1).numpy()
    concept_label_strings = create_labels(concept_outputs, no_of_categories=no_of_categories)
    visualization = show_factorization_on_image(rgb_img_float,
                                                batch_explanations[0],
                                                image_weight=0.5,
                                                concept_labels=concept_label_strings)
    result = np.hstack((img, visualization))
    return result


# https://jacobgil.github.io/pytorch-gradcam-book/Deep%20Feature%20Factorizations.html
def visualize_image2(input_model, input_image, vis_input_tensor, no_of_components=1, no_of_categories=1):
    img, rgb_img_float = get_image_from_file(input_image)
    classifier = input_model.fc
    dff = DeepFeatureFactorization(model=input_model, target_layer=input_model.layer4, computation_on_concepts=classifier)
    concepts, batch_explanations, concept_outputs = dff(vis_input_tensor, no_of_components)
    concept_outputs = torch.softmax(torch.from_numpy(concept_outputs), dim=-1).numpy()
    concept_label_strings = create_labels2(concept_outputs, no_of_categories=no_of_categories)
    # modified_concept_labels = [label.replace(" ", "\n") if " " in label else label for label in concept_label_strings]
    visualization = show_factorization_on_image(rgb_img_float,
                                                batch_explanations[0],
                                                image_weight=0.5,
                                                concept_labels=concept_label_strings)
    result = np.hstack((img, visualization))
    return result


def train_and_save_model(input_model, train_dataset, val_data_subset):
    for parameter in input_model.parameters():
        parameter.requires_grad = False

    for parameter in input_model.fc.parameters():
        parameter.requires_grad = True

    train_batch_sampler = BalancedBatchSampler(train_dataset, 10, 10)
    val_batch_sampler = BalancedBatchSampler(val_data_subset, 10, 10)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler)
    val_loader = torch.utils.data.DataLoader(val_data_subset, batch_sampler=val_batch_sampler)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(input_model.parameters(), lr=1e-4)

    num_epochs = 3
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        input_model.train()
        running_loss = 0.0

        for batch_idx, (inputs, tl_labels) in enumerate(tqdm(train_loader), 1):
            optimizer.zero_grad()
            outputs = input_model(inputs)
            loss = criterion(outputs, tl_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)

        input_model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, vl_labels) in enumerate(tqdm(val_loader), 1):
                outputs = input_model(inputs)
                _, predicted = torch.max(outputs, 1)
                vl_labels = torch.tensor(vl_labels)
                total += vl_labels.size(0)
                correct += (predicted == vl_labels).sum().item()

        epoch_accuracy = correct / total
        print(f"Training Loss: {epoch_loss:.4f}, Validation Accuracy: {epoch_accuracy:.2%}")

        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(input_model.state_dict(), "best_model.pth")
            print("Model saved as best_model.pth")

    print("Training completed.")
    exit(0)


def evaluate_model(input_model, data_loader):
    input_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, dl_labels in data_loader:
            outputs = input_model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu())
            all_labels.append(dl_labels.cpu())
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        return all_preds, all_labels


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# streamlit run C:/Users/Marcus/PycharmProjects/pythonProject1/main.py

st.set_page_config(layout="wide")
_, center, _ = st.columns([1, 8, 1])  # workaround for open issue @ https://github.com/streamlit/streamlit/issues/5466
plt.rcParams['figure.dpi'] = 82
torch.manual_seed(123)
warnings.filterwarnings('ignore')


@st.cache_resource
def load_resources():
    resize_transform = transforms.Resize((320, 320))
    tensor_transform = transforms.ToTensor()
    composed_transform = transforms.Compose([resize_transform, tensor_transform])

    l_map = {
        0: 0,  # tench
        1: 217,  # English springer
        2: 482,  # cassette player
        3: 491,  # chain saw
        4: 497,  # church
        5: 566,  # French horn
        6: 569,  # garbage truck
        7: 571,  # gas pump
        8: 574,  # golf ball
        9: 701  # parachute
    }

    t_data = Imagenette(root="Imagenette/train", split="train", size="320px", transform=composed_transform)  # 9469
    v_data = Imagenette(root="Imagenette/val", split="val", size="320px", transform=resize_transform)  # 3925

    lengths = [int(len(v_data) * 0.6), int(len(v_data) * 0.4)]
    v_subset, h_subset = random_split(v_data, lengths)
    v_subset = DatasetFromSubset(v_subset, transform=composed_transform)
    h_subset1 = DatasetFromSubset(h_subset, transform=resize_transform)
    h_subset2 = DatasetFromSubset(h_subset, transform=composed_transform)

    return {
        'label_map': l_map,
        'train_data': t_data,
        'val_subset': v_subset,
        'holdout_subset1': h_subset1,
        'holdout_subset2': h_subset2
    }


resources = load_resources()

label_map = resources['label_map']
train_data = resources['train_data']
val_subset = resources['val_subset']
holdout_subset1 = resources['holdout_subset1']
holdout_subset2 = resources['holdout_subset2']

with center:
    tab1, tab2 = st.tabs(["Fine-tuned", "Pretrained"])

    col1, col2, col3 = st.columns(3)
    with col1:
        image_index = st.number_input('index', min_value=0, step=1, value=0, format='%d')
    with col2:
        n_components = st.number_input('n comp', min_value=1, step=1, value=1, format='%d')
    with col3:
        top_k = st.number_input('top k', min_value=1, step=1, value=1, format='%d')

    image, label = holdout_subset1[image_index]
    rgb_img = np.float32(image) / 255
    input_tensor, _ = holdout_subset2[image_index]
    input_tensor = input_tensor.unsqueeze(0)

    # ------------------------------------------------------------------------------------------------------------------

    with tab1:
        model = resnet18(pretrained=True)
        model.fc = nn.Linear(512, 10)  # fc is the name of the classification layer

        if not os.path.exists("best_model.pth"):
            train_and_save_model(model, train_data, val_subset)

        model.load_state_dict(torch.load("best_model.pth"))
        model.eval()

        for param in model.parameters():
            param.requires_grad = True

        outputs_hirescam = model(input_tensor)
        probabilities_hirescam = torch.nn.functional.softmax(outputs_hirescam, dim=1)
        top1_prob, top1_index = torch.topk(probabilities_hirescam, 1, dim=1)
        top1_prob = top1_prob.detach().numpy().item()
        top1_index = label_map[torch.Tensor.numpy(top1_index).item()]
        cam = HiResCAM(model=model, target_layers=[model.layer4[-1]])
        grayscale_cam = cam(input_tensor=input_tensor)
        grayscale_cam = grayscale_cam[0, :]
        visualization_hirescam = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        for param in model.parameters():
            param.requires_grad = False

        ima_dff = visualize_image(model, image, input_tensor, no_of_components=n_components, no_of_categories=top_k)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 3]})
        ax1.imshow(visualization_hirescam)
        ax1.axis('off')
        ax1.set_title("HiResCAM")
        ax2.imshow(ima_dff)
        ax2.axis('off')
        ax2.set_title("DFF")
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()

        if image is not None:
            st.pyplot(fig, use_container_width=True)
            col_left, col1, col2, col_right = st.columns([2, 2, 2, 1])

            with col1:
                st.write("Predicted class:")
                st.write(f"{get_label_from_numeric(top1_index)}: {round(top1_prob, 2)}")

            with col2:
                st.write("True class:\n")
                st.write(get_label_from_numeric(label_map[label]))

    # ------------------------------------------------------------------------------------------------------------------

    with tab2:
        model = resnet18(pretrained=True)
        model.eval()

        for param in model.parameters():
            param.requires_grad = True

        outputs_hirescam = model(input_tensor)
        probabilities_hirescam = torch.nn.functional.softmax(outputs_hirescam, dim=1)
        _, predicted_indices_hirescam = torch.topk(probabilities_hirescam, 3)
        predicted_indices_hirescam = predicted_indices_hirescam.squeeze().tolist()
        probabilities_np_hirescam = probabilities_hirescam.detach().numpy()
        concept_labels_hirescam = create_labels2(probabilities_np_hirescam, 1)
        cam = HiResCAM(model=model, target_layers=[model.layer4[-1]])
        grayscale_cam = cam(input_tensor=input_tensor)
        grayscale_cam = grayscale_cam[0, :]
        visualization_hirescam = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        for param in model.parameters():
            param.requires_grad = False

        ima_dff = visualize_image2(model, image, input_tensor, no_of_components=n_components, no_of_categories=top_k)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 3]})
        ax1.imshow(visualization_hirescam)
        ax1.axis('off')
        ax1.set_title("HiResCAM")
        ax2.imshow(ima_dff)
        ax2.axis('off')
        ax2.set_title("DFF")
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()

        if image is not None:
            st.pyplot(fig, use_container_width=True)
            col_left, col1, col2, col_right = st.columns([2, 2, 2, 1])

            with col1:
                st.write("Predicted class:")
                for _, labels in enumerate(concept_labels_hirescam[0].split("\n"), 1):
                    labels = labels.replace(":", ": ")
                    st.write(f"{labels}")

            with col2:
                st.write("True class:\n")
                st.write(get_label_from_numeric(label_map[label]))

    # loader = torch.utils.data.DataLoader(val_subset, batch_size=100, shuffle=True)
    # val_preds, val_labels = evaluate_model(model, loader)
    # val_conf_matrix = confusion_matrix(val_labels, val_preds)
    # val_bal_acc = balanced_accuracy_score(val_labels, val_preds)
    # print("Validation Confusion Matrix:\n", val_conf_matrix)
    # print("Validation Balanced Accuracy: ", val_bal_acc)
