# 第二十二章：神经网络用于非结构化数据

# 22.0 介绍

在前一章中，我们专注于适用于*结构化*数据的神经网络配方，即表格数据。实际上，过去几年中最大的进展大部分涉及使用神经网络和深度学习处理*非结构化*数据，例如文本或图像。与处理结构化数据源不同，处理这些非结构化数据集有所不同。

深度学习在非结构化数据领域尤为强大，而“经典”的机器学习技术（如提升树）通常无法捕捉文本数据、音频、图像、视频等中存在的所有复杂性和细微差别。在本章中，我们将专门探讨将深度学习用于文本和图像数据。

在文本和图像的监督学习空间中，存在许多子任务或“类型”学习。以下是一些示例（尽管这不是一个全面的列表）：

+   文本或图像分类（例如：分类一张照片是否是热狗的图像）

+   迁移学习（例如：使用预训练的上下文模型如 BERT，并在一个任务上进行微调以预测电子邮件是否为垃圾邮件）

+   目标检测（例如：识别和分类图像中的特定对象）

+   生成模型（例如：基于给定输入生成文本的模型，如 GPT 模型）

随着深度学习的普及和越来越普遍，处理这些用例的开源和企业解决方案变得更加易于访问。在本章中，我们将利用几个关键库作为我们进入执行这些深度学习任务的起点。特别是，我们将使用 PyTorch、Torchvision 和 Transformers Python 库来完成跨文本和图像 ML 数据的一系列任务。

# 22.1 训练神经网络进行图像分类

## 问题

您需要训练一个图像分类神经网络。

## 解决方案

在 PyTorch 中使用卷积神经网络：

```py
# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Define the convolutional neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(self.dropout1(x), 2)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(self.dropout2(x)))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

# Set the device to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the data preprocessing steps
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the MNIST dataset
train_dataset = datasets.MNIST('./data', train=True, download=True,
    transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

# Create data loaders
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
    shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
    shuffle=True)

# Initialize the model and optimizer
model = Net().to(device)
optimizer = optim.Adam(model.parameters())

# Compile the model using torch 2.0's optimizer
model = torch.compile(model)

# Define the training loop
model.train()
for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = nn.functional.nll_loss(output, target)
    loss.backward()
    optimizer.step()

# Define the testing loop
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)

        # get the index of the max log-probability
        test_loss += nn.functional.nll_loss(
            output, target, reduction='sum'
        ).item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)
```

## 讨论

卷积神经网络通常用于图像识别和计算机视觉任务。它们通常包括卷积层、池化层和全连接层。

*卷积层*的目的是学习可以用于当前任务的重要图像特征。卷积层通过对图像的特定区域（卷积的大小）应用滤波器来工作。这一层的权重然后学习识别在分类任务中关键的特定图像特征。例如，如果我们在训练一个识别人手的模型，滤波器可能学会识别手指。

*池化层*的目的通常是从前一层的输入中减少维度。该层还使用应用于输入部分的滤波器，但没有激活。相反，它通过执行*最大池化*（选择具有最高值的滤波器中的像素）或*平均池化*（取输入像素的平均值来代替）来减少输入的维度。

最后，*全连接层*可以与类似 softmax 的激活函数一起用于创建一个二元分类任务。

## 参见

+   [卷积神经网络](https://oreil.ly/HoO9g)

# 22.2 训练用于文本分类的神经网络

## 问题

您需要训练一个神经网络来对文本数据进行分类。

## 解决方案

使用一个 PyTorch 神经网络，其第一层是您的词汇表的大小：

```py
# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the 20 newsgroups dataset
cats = ['alt.atheism', 'sci.space']
newsgroups_data = fetch_20newsgroups(subset='all', shuffle=True,
    random_state=42, categories=cats)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(newsgroups_data.data,
    newsgroups_data.target, test_size=0.2, random_state=42)

# Vectorize the text data using a bag-of-words approach
vectorizer = CountVectorizer(stop_words='english')
X_train = vectorizer.fit_transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Define the model
class TextClassifier(nn.Module):
    def __init__(self, num_classes):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

# Instantiate the model and define the loss function and optimizer
model = TextClassifier(num_classes=len(cats))
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Compile the model using torch 2.0's optimizer
model = torch.compile(model)

# Train the model
num_epochs = 1
batch_size = 10
num_batches = len(X_train) // batch_size
for epoch in range(num_epochs):
    total_loss = 0.0
    for i in range(num_batches):
        # Prepare the input and target data for the current batch
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        inputs = X_train[start_idx:end_idx]
        targets = y_train[start_idx:end_idx]

        # Zero the gradients for the optimizer
        optimizer.zero_grad()

        # Forward pass through the model and compute the loss
        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        # Backward pass through the model and update the parameters
        loss.backward()
        optimizer.step()

        # Update the total loss for the epoch
        total_loss += loss.item()

    # Compute the accuracy on the test set for the epoch
    test_outputs = model(X_test)
    test_predictions = torch.argmax(test_outputs, dim=1)
    test_accuracy = accuracy_score(y_test, test_predictions)

    # Print the epoch number, average loss, and test accuracy
    print(f"Epoch: {epoch+1}, Loss: {total_loss/num_batches}, Test Accuracy:"
        "{test_accuracy}")
```

## 讨论

不像图像，文本数据本质上是非数值的。在训练模型之前，我们需要将文本转换为模型可以使用的数值表示，以便学习哪些单词和单词组合对于当前分类任务是重要的。在这个例子中，我们使用 scikit-learn 的`CountVectorizer`将词汇表编码为一个大小为整个词汇表的向量，其中每个单词被分配到向量中的特定索引，该位置的值是该单词在给定段落中出现的次数。在这种情况下，我们可以通过查看我们的训练集来看到词汇表的大小：

```py
X_train.shape[1]
```

```py
25150
```

我们在神经网络的第一层使用相同的值来确定输入层的大小：`self.fc1 = nn.Linear(X_train.shape[1], 128)`。这允许我们的网络学习所谓的*词嵌入*，即从像本配方中的监督学习任务学习到的单词的向量表示。这个任务将允许我们学习大小为 128 的词嵌入，尽管这些嵌入主要对这个特定的任务和词汇表有用。

# 22.3 对图像分类进行微调预训练模型

## 问题

您希望使用从预训练模型中学到的知识来训练图像分类模型。

## 解决方案

使用`transformers`库和`torchvision`在您的数据上对预训练模型进行微调：

```py
# Import libraries
import torch
from torchvision.transforms import(
    RandomResizedCrop, Compose, Normalize, ToTensor
    )
from transformers import Trainer, TrainingArguments, DefaultDataCollator
from transformers import ViTFeatureExtractor, ViTForImageClassification
from datasets import load_dataset, load_metric, Image

# Define a helper function to convert the images into RGB
def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in
        examples["image"]]
    del examples["image"]
    return examples

# Define a helper function to compute metrics
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1),
        references=p.label_ids)

# Load the fashion mnist dataset
dataset = load_dataset("fashion_mnist")

# Load the processor from the VIT model
image_processor = ViTFeatureExtractor.from_pretrained(
    "google/vit-base-patch16-224-in21k"
)

# Set the labels from the dataset
labels = dataset['train'].features['label'].names

# Load the pretrained model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(labels),
    id2label={str(i): c for i, c in enumerate(labels)},
    label2id={c: str(i) for i, c in enumerate(labels)}
)

# Define the collator, normalizer, and transforms
collate_fn = DefaultDataCollator()
normalize = Normalize(mean=image_processor.image_mean,
    std=image_processor.image_std)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

# Load the dataset we'll use with transformations
dataset = dataset.with_transform(transforms)

# Use accuracy as our metric
metric = load_metric("accuracy")

# Set the training args
training_args = TrainingArguments(
    output_dir="fashion_mnist_model",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=0.01,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)

# Instantiate a trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=image_processor,
)

# Train the model, log and save metrics
train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()
```

## 讨论

在像文本和图像这样的非结构化数据领域，通常会使用在大型数据集上训练过的预训练模型作为起点，而不是从头开始，尤其是在我们没有太多标记数据的情况下。利用来自更大模型的嵌入和其他信息，我们可以调整我们自己的模型以适应新任务，而不需要大量标记信息。此外，预训练模型可能具有我们训练数据中未完全捕获的信息，从而导致整体性能的提升。这个过程被称为*迁移学习*。

在这个例子中，我们加载了来自 Google 的 ViT（Vision Transformer）模型的权重。然后，我们使用`transformers`库对其进行微调，以在时尚 MNIST 数据集上进行分类任务，这是一个简单的服装项目数据集。这种方法可以应用于增加任何计算机视觉数据集的性能，并且`transformers`库提供了一个高级接口，我们可以使用它来微调我们自己的模型，而无需编写大量代码。

## 参见

+   [Hugging Face 网站和文档](https://oreil.ly/5F3Rf)

# 22.4 对预训练模型进行文本分类的微调

## 问题

你想使用预训练模型的学习成果来训练一个文本分类模型。

## 解决方案

使用`transformers`库：

```py
# Import libraries
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import (
    AutoModelForSequenceClassification, TrainingArguments, Trainer
    )
import evaluate
import numpy as np

# Load the imdb dataset
imdb = load_dataset("imdb")

# Create a tokenizer and collator
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Tokenize the imdb dataset
tokenized_imdb = imdb.map(
    lambda example: tokenizer(
        example["text"], padding="max_length", truncation=True
    ),
    batched=True,
)

# User the accuracy metric
accuracy = evaluate.load("accuracy")

# Define a helper function to produce metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

# Create dictionaries to map indices to labels and vice versa
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

# Load a pretrained model
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label,
        label2id=label2id
)

# Specify the training arguments
training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# Instantiate a trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()
```

## 讨论

就像使用预训练图像模型一样，预训练语言模型包含了大量关于语言的上下文信息，因为它们通常是在各种开放互联网来源上进行训练的。当我们从一个预训练模型基础开始时，我们通常做的是将现有网络的分类层替换为我们自己的分类层。这使我们能够修改已经学习的网络权重以适应我们的特定任务。

在这个例子中，我们正在对一个 DistilBERT 模型进行微调，以识别 IMDB 电影评论是积极的（1）还是消极的（0）。预训练的 DistilBERT 模型为每个单词提供了大量的语境信息，以及从先前的训练任务中学到的神经网络权重。迁移学习使我们能够利用所有用于训练 DistilBERT 模型的初始工作，并将其重新用于我们的用例，即对电影评论进行分类。

## 参见

+   [transformers 中的文本分类](https://oreil.ly/uhrjI)
