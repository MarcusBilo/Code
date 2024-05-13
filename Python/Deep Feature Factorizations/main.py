# pip install grad-cam
import warnings
import numpy as np
import requests
import torch
import torchvision
from pytorch_grad_cam import DeepFeatureFactorization
from pytorch_grad_cam.utils.image import preprocess_image, show_factorization_on_image
from torchvision.models import resnet50, resnet18
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from datasets import load_dataset
import os
import torch.nn as nn
import torch.optim as optim
import pickle
from torchvision.datasets import Imagenette
from tqdm import tqdm


def get_image_from_file(image):
    img = np.array(image)
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
                                                image_weight=0.3,
                                                concept_labels=concept_label_strings)

    result = np.hstack((img, visualization))
    return result


# ----------------------------------------------------------------------------------------------------------------------

warnings.filterwarnings('ignore')

if not os.path.exists("best_model.pth"):
    # Load pre-trained ResNet model
    model = resnet18(pretrained=True)

    # Freeze the parameters in the pre-trained layers except the last few layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last few layers for fine-tuning
    for param in model.layer4.parameters():
        param.requires_grad = True

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_data = Imagenette(root="Imagenette/train", split="train", size="320px", transform=transform)  # 9469
    val_data = Imagenette(root="Imagenette/val", split="val", size="320px", transform=transform)  # 3925

    # Set batch size and create data loaders
    batch_size = 100
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    num_epochs = 2
    best_accuracy = 0.0

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Set model to training mode
        model.train()

        # Initialize running loss for this epoch
        running_loss = 0.0

        # Iterate over training data
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader), 1):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate the running loss
            running_loss += loss.item() * inputs.size(0)

            # Print mini-batch progress
            if batch_idx % 100 == 0:
                print(f"    Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Calculate average training loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate accuracy on validation set
        epoch_accuracy = correct / total

        # Print epoch statistics
        print(f"    Training Loss: {epoch_loss:.4f}, Validation Accuracy: {epoch_accuracy:.2%}")

        # Save the model if it performs better than the previous best
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), "best_model.pth")
            print("    Model saved as best_model.pth")

    print("Training completed.")

model = resnet18(pretrained=True)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((320, 320)),
])

train_data = Imagenette(root="Imagenette/train", split="train", size="320px", transform=transform)  # 9469
val_data = Imagenette(root="Imagenette/val", split="val", size="320px", transform=transform)  # 3925

image, _ = train_data[0]
_, _, input_tensor = get_image_from_file(image)

with torch.no_grad():
    model.eval()
    outputs = model(input_tensor)

predicted_classes = create_labels_v2(outputs.numpy(), top_k=10)

print("Top predicted classes:")
for i, labels in enumerate(predicted_classes[0].split("\n"), 1):
    print(f"{i}. {labels}")

f0 = plt.figure(0)

ima = visualize_image(model, image, n_components=3, top_k=3)
plt.imshow(ima)
plt.show()
