import torch
from torchvision import models
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import numpy as np

from tqdm import tqdm
from matplotlib import pyplot as plt
import pickle
import faiss
from PIL import Image
import io

from app.constants import K, DEVICE, DEVICE_CPU


def get_model():
    # Загрузка модели ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Убираем последний полносвязный слой (FC)
    model.fc = torch.nn.Identity()

    # Выключаем вычисление градиентов и включаем режим оценки
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    model = model.to(DEVICE)
    return model


# Трансформация изображений
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(
            num_output_channels=3
        ),  # Теперь делаем изображение трёхканальным
        transforms.ToTensor(),
    ]
)

# Загрузка FashionMNIST
dataset = FashionMNIST(root="./data", train=True, download=True, transform=transform)

with open("./data/fashionmnist_features.pkl", "rb") as f:
    all_features = pickle.load(f)
norm_all_features = all_features / np.linalg.norm(all_features, axis=1, keepdims=True)
index = faiss.IndexFlatL2(norm_all_features.shape[1])
index.add(norm_all_features)

model = get_model()


def tensor_to_pil(inp):
    """Конвертирует тензор в PIL Image"""
    if isinstance(inp, torch.Tensor):
        # Денормализуем если нужно и конвертируем в numpy
        inp = inp.permute(1, 2, 0).numpy()
        # Масштабируем от 0 до 255 если нужно
        if inp.max() <= 1.0:
            inp = (inp * 255).astype(np.uint8)
        else:
            inp = inp.astype(np.uint8)

    return Image.fromarray(inp)


def matplotlib_figure_to_bytes(fig):
    """
    Конвертирует matplotlib figure в bytes

    Args:
        fig: matplotlib.figure.Figure

    Returns:
        bytes: изображение в формате PNG
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    return buf.getvalue()


def create_matplotlib_grid(pil_images, labels=None, rows=3, cols=3, figsize=(12, 12)):
    """
    Создает сетку изображений как matplotlib figure

    Args:
        pil_images: List[PIL.Image] - список PIL Images
        labels: List[str] - список подписей для изображений
        rows: int - количество строк
        cols: int - количество колонок
        figsize: tuple - размер фигуры (width, height)

    Returns:
        plt.Figure: matplotlib figure object
    """
    num_images = min(len(pil_images), rows * cols)

    # Создаем фигуру и оси
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Если только одно изображение, преобразуем axes в массив
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1 or cols == 1:
        axes = axes.reshape(-1)

    # Скрываем лишние оси
    for i in range(num_images, rows * cols):
        if rows > 1 and cols > 1:
            axes[i // cols, i % cols].axis("off")
        else:
            axes[i].axis("off")

    # Отображаем изображения
    for idx in range(num_images):
        if rows > 1 and cols > 1:
            ax = axes[idx // cols, idx % cols]
        else:
            ax = axes[idx]

        # Конвертируем PIL Image в numpy array для matplotlib
        img_array = np.array(pil_images[idx])

        # Отображаем изображение
        ax.imshow(img_array)
        ax.axis("off")  # Скрываем оси

        # Добавляем подпись если есть
        if labels and idx < len(labels):
            ax.set_title(labels[idx], fontsize=12, pad=10)

    # Настраиваем расстояние между subplots
    plt.tight_layout()

    return fig


# Полный процесс: от тензоров до matplotlib figure
def create_matplotlib_from_tensors(images_tensors, labels=None, rows=3, cols=3):
    """
    Полный процесс: конвертирует тензоры в matplotlib figure
    """
    # Конвертируем тензоры в PIL Images
    pil_images = [tensor_to_pil(img) for img in images_tensors]

    # Создаем matplotlib figure
    fig = create_matplotlib_grid(pil_images, labels, rows, cols)

    return fig


def process_one_image_from_model(img):
    with torch.no_grad():
        image = img.to(DEVICE)
        outputs = model(image)
        image = image.to(DEVICE_CPU)
        image_after_model = outputs.cpu().numpy()
        return image_after_model


def search_nearest_neighbors(img):
    """
    Функция поиска ближайших соседей.
    :param query_vector: Вектор-запрос (предварительно нормализованный!)
    :param k: Количество ближайших соседей
    :return: индексы ближайших соседей и расстояния до них
    """
    query_vector = img.reshape(1, -1)

    distances, indices = index.search(query_vector.reshape(1, -1), K)
    return indices.flatten(), distances.flatten()


def preprocess_image(image):
    image_data = io.BytesIO(image)
    image = Image.open(image_data)  # Открытие изображения
    transformed_image = transform(image)  # Применяем трансформации
    return transformed_image.unsqueeze(0)


def process_image_and_preview(image):
    transformed_img_image = preprocess_image(image)

    image_after_model = process_one_image_from_model(transformed_img_image)

    nearest_indices, nearest_distances = search_nearest_neighbors(image_after_model)

    images = []
    labels = []

    # nearest_indices = [0, 1, 2]

    for i in nearest_indices:
        images.append(dataset[i][0])
        labels.append(dataset[i][1])

    # Сохраняем сетку как PIL Image
    grid_image = create_matplotlib_from_tensors(images, labels)

    image_bytes = matplotlib_figure_to_bytes(grid_image)

    return image_bytes
