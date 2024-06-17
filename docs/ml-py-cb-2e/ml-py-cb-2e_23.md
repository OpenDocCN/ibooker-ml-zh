# 第二十三章\. 保存、加载和提供训练好的模型

# 23.0 简介

在过去的 22 章和大约 200 个示例中，我们已经涵盖了如何使用机器学习从原始数据创建性能良好的预测模型。然而，为了使我们所有的工作变得有价值，最终我们需要*对我们的模型采取行动*，比如将其集成到现有的软件应用程序中。为了实现这个目标，我们需要能够在训练后保存我们的模型，在应用程序需要时加载它们，然后请求该应用程序获取预测结果。

机器学习模型通常部署在简单的 Web 服务器上，旨在接收输入数据并返回预测结果。这使得模型能够在同一网络上的任何客户端中使用，因此其他服务（如 UI、用户等）可以实时使用 ML 模型进行预测。例如，在电子商务网站上使用 ML 进行商品搜索时，将提供一个 ML 模型，该模型接收关于用户和列表的数据，并返回用户购买该列表的可能性。搜索结果需要实时可用，并且可供负责接收用户搜索并协调用户结果的电子商务应用程序使用。

# 23.1 保存和加载 scikit-learn 模型

## 问题

您有一个训练好的 scikit-learn 模型，想要在其他地方保存和加载它。

## 解决方案

将模型保存为 pickle 文件：

```py
# Load libraries
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

# Load data
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Create decision tree classifer object
classifer = RandomForestClassifier()

# Train model
model = classifer.fit(features, target)

# Save model as pickle file
joblib.dump(model, "model.pkl")
```

```py
['model.pkl']
```

一旦模型保存完成，我们可以在目标应用程序（例如 Web 应用程序）中使用 scikit-learn 加载该模型：

```py
# Load model from file
classifer = joblib.load("model.pkl")
```

并使用它进行预测：

```py
# Create new observation
new_observation = [[ 5.2,  3.2,  1.1,  0.1]]

# Predict observation's class
classifer.predict(new_observation)
```

```py
array([0])
```

## 讨论

将模型用于生产环境的第一步是将该模型保存为可以被另一个应用程序或工作流加载的文件。我们可以通过将模型保存为 pickle 文件来实现这一点，pickle 是一种 Python 特定的数据格式，使我们能够序列化 Python 对象并将其写入文件。具体来说，为了保存模型，我们使用 `joblib`，这是一个扩展 pickle 的库，用于处理我们在 scikit-learn 中经常遇到的大型 NumPy 数组。

在保存 scikit-learn 模型时，请注意保存的模型可能在不同版本的 scikit-learn 之间不兼容；因此，在文件名中包含使用的 scikit-learn 版本可能会有所帮助：

```py
# Import library
import sklearn

# Get scikit-learn version
scikit_version = sklearn.__version__

# Save model as pickle file
joblib.dump(model, "model_{version}.pkl".format(version=scikit_version))
```

```py
['model_1.2.0.pkl']
```

# 23.2 保存和加载 TensorFlow 模型

## 问题

您有一个训练好的 TensorFlow 模型，想要在其他地方保存和加载它。

## 解决方案

使用 TensorFlow 的 `saved_model` 格式保存模型：

```py
# Load libraries
import numpy as np
from tensorflow import keras

# Set random seed
np.random.seed(0)

# Create model with one hidden layer
input_layer = keras.Input(shape=(10,))
hidden_layer = keras.layers.Dense(10)(input_layer)
output_layer = keras.layers.Dense(1)(input_layer)
model = keras.Model(input_layer, output_layer)
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
x_train = np.random.random((1000, 10))
y_train = np.random.random((1000, 1))
model.fit(x_train, y_train)

# Save the model to a directory called `save_model`
model.save("saved_model")
```

```py
32/32 [==============================] - 1s 8ms/step - loss: 0.2056
INFO:tensorflow:Assets written to: saved_model/assets
```

然后我们可以在另一个应用程序中加载该模型，或用于进一步的训练：

```py
# Load neural network
model =  keras.models.load_model("saved_model")
```

## 讨论

虽然在本书的整个过程中我们并没有大量使用 TensorFlow，但了解如何保存和加载 TensorFlow 模型仍然是有用的。与使用 Python 原生的 `pickle` 格式不同，TensorFlow 提供了自己的保存和加载模型的方法。`saved_model` 格式创建一个存储模型和加载所需所有信息的目录，以便以协议缓冲区格式（使用 *.pb* 文件扩展名）加载模型并进行预测：

```py
ls saved_model
```

```py
assets	fingerprint.pb	keras_metadata.pb  saved_model.pb  variables
```

虽然我们不会深入探讨这种格式，但这是在 TensorFlow 中保存、加载和提供训练模型的标准方式。

## 参见

+   [序列化和保存 Keras 模型](https://oreil.ly/CDPvo)

+   [TensorFlow 保存模型格式](https://oreil.ly/StpSL)

# 23.3 保存和加载 PyTorch 模型

## 问题

如果你有一个训练好的 PyTorch 模型，并希望在其他地方保存和加载它。

## 解决方案

使用 `torch.save` 和 `torch.load` 函数：

```py
# Load libraries
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import RMSprop
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create training and test sets
features, target = make_classification(n_classes=2, n_features=10,
    n_samples=1000)
features_train, features_test, target_train, target_test = train_test_split(
    features, target, test_size=0.1, random_state=1)

# Set random seed
torch.manual_seed(0)
np.random.seed(0)

# Convert data to PyTorch tensors
x_train = torch.from_numpy(features_train).float()
y_train = torch.from_numpy(target_train).float().view(-1, 1)
x_test = torch.from_numpy(features_test).float()
y_test = torch.from_numpy(target_test).float().view(-1, 1)

# Define a neural network using `Sequential`
class SimpleNeuralNet(nn.Module):
    def __init__(self):
        super(SimpleNeuralNet, self).__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(10, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Dropout(0.1), # Drop 10% of neurons
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.sequential(x)
        return x

# Initialize neural network
network = SimpleNeuralNet()

# Define loss function, optimizer
criterion = nn.BCELoss()
optimizer = RMSprop(network.parameters())

# Define data loader
train_data = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_data, batch_size=100, shuffle=True)

# Compile the model using torch 2.0's optimizer
network = torch.compile(network)

# Train neural network
epochs = 5
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Save the model after it's been trained
torch.save(
    {
        'epoch': epoch,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    },
    "model.pt"
)

# Reinitialize neural network
network = SimpleNeuralNet()
state_dict = torch.load(
    "model.pt",
    map_location=torch.device('cpu')
    )["model_state_dict"]
network.load_state_dict(state_dict, strict=False)
network.eval()
```

```py
SimpleNeuralNet(
  (sequential): Sequential(
    (0): Linear(in_features=10, out_features=16, bias=True)
    (1): ReLU()
    (2): Linear(in_features=16, out_features=16, bias=True)
    (3): ReLU()
    (4): Linear(in_features=16, out_features=1, bias=True)
    (5): Dropout(p=0.1, inplace=False)
    (6): Sigmoid()
  )
)
```

## 讨论

尽管我们在 第二十一章 中使用了类似的公式来检查点我们的训练进度，但在这里我们看到相同的方法如何用于将模型加载回内存以进行预测。我们保存模型的 `model.pt` 实际上只是一个包含模型参数的字典。我们在字典键 `model_state_dict` 中保存了模型状态；为了将模型加载回来，我们重新初始化我们的网络，并使用 `network.load_state_dict` 加载模型的状态。

## 参见

+   [PyTorch 教程：保存和加载模型](https://oreil.ly/WO3X1)

# 23.4 提供 scikit-learn 模型

## 问题

你想要使用 Web 服务器提供你训练好的 scikit-learn 模型。

## 解决方案

构建一个 Python Flask 应用程序，加载本章早期训练的模型：

```py
# Import libraries
import joblib
from flask import Flask, request

# Instantiate a flask app
app = Flask(__name__)

# Load the model from disk
model = joblib.load("model.pkl")

# Create a predict route that takes JSON data, makes predictions, and
# returns them
@app.route("/predict", methods = ["POST"])
def predict():
    print(request.json)
    inputs = request.json["inputs"]
    prediction = model.predict(inputs)
    return {
        "prediction" : prediction.tolist()
    }

 # Run the app
if __name__ == "__main__":
    app.run()
```

确保已安装 Flask：

```py
python3 -m pip install flask==2.2.3 joblib==1.2.0 scikit-learn==1.2.0
```

然后运行应用程序：

```py
python3 app.py
```

```py
 * Serving Flask app 'app'
 * Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment.
    Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
```

现在，我们可以通过向端点提交数据点来对应用程序进行预测，并通过 `curl` 获取结果：

```py
curl -X POST http://127.0.0.1:5000/predict  -H 'Content-Type: application/json'
    -d '{"inputs":[[5.1, 3.5, 1.4, 0.2]]}'
```

```py
{"prediction":[0]}
```

## 讨论

在本例中，我们使用了 Flask，这是一个流行的用于构建 Python Web 框架的开源库。我们定义了一个路由 `/predict`，该路由接受 POST 请求中的 JSON 数据，并返回包含预测结果的字典。尽管这个服务器并非准备用于生产环境（请参阅 Flask 关于使用开发服务器的警告），我们可以很容易地使用更适合生产环境的 Web 框架扩展和提供此代码以将其移至生产环境。

# 23.5 提供 TensorFlow 模型

## 问题

你想要使用 Web 服务器提供你训练好的 TensorFlow 模型。

## 解决方案

使用开源 TensorFlow Serving 框架和 Docker：

```py
docker run -p 8501:8501 -p 8500:8500 \
--mount type=bind,source=$(pwd)/saved_model,target=/models/saved_model/1 \
-e MODEL_NAME=saved_model -t tensorflow/serving
```

## 讨论

TensorFlow Serving 是一个针对 TensorFlow 模型优化的开源服务解决方案。通过简单提供模型路径，我们就能获得一个 HTTP 和 gRPC 服务器，并附带开发者所需的额外有用功能。

`docker run` 命令使用公共 `tensorflow/serving` 镜像运行容器，并将我们当前工作目录的 `saved_model` 路径 (`$(pwd)/saved_model`) 挂载到容器内部的 `/models/saved_model/1`。这样就会自动将我们之前在本章保存的模型加载到正在运行的 Docker 容器中，我们可以向其发送预测查询。

如果你在网络浏览器中转到 [*http://localhost:8501/v1/models/saved_model*](http://localhost:8501/v1/models/saved_model)，你应该看到这里显示的 JSON 结果：

```py
{
    "model_version_status": [
        {
            "version": "1",
            "state": "AVAILABLE",
            "status": {
                "error_code": "OK",
                "error_message": ""
            }
        }
    ]
}
```

在 [*http://localhost:8501/v1/models/saved_model/metadata*](http://localhost:8501/v1/models/saved_model/metadata) 的 `/metadata` 路由将返回有关模型的更多信息：

```py
{
"model_spec":{
 "name": "saved_model",
 "signature_name": "",
 "version": "1"
}
,
"metadata": {"signature_def": {
 "signature_def": {
  "serving_default": {
   "inputs": {
    "input_8": {
     "dtype": "DT_FLOAT",
     "tensor_shape": {
      "dim": [
       {
        "size": "-1",
        "name": ""
       },
       {
        "size": "10",
        "name": ""
       }
      ],
      "unknown_rank": false
     },
     "name": "serving_default_input_8:0"
    }
   },
   "outputs": {
    "dense_11": {
     "dtype": "DT_FLOAT",
     "tensor_shape": {
      "dim": [
       {
        "size": "-1",
        "name": ""
       },
       {
        "size": "1",
        "name": ""
       }
      ],
      "unknown_rank": false
     },
     "name": "StatefulPartitionedCall:0"
    }
   },
   "method_name": "tensorflow/serving/predict"
  },
  "__saved_model_init_op": {
   "inputs": {},
   "outputs": {
    "__saved_model_init_op": {
     "dtype": "DT_INVALID",
     "tensor_shape": {
      "dim": [],
      "unknown_rank": true
     },
     "name": "NoOp"
    }
   },
   "method_name": ""
  }
 }
}
}
}
```

我们可以使用 `curl` 向 REST 端点进行预测，并传递变量（此神经网络使用 10 个特征）：

```py
curl -X POST http://localhost:8501/v1/models/saved_model:predict
    -d '{"inputs":[[1,2,3,4,5,6,7,8,9,10]]}'
```

```py
{
    "outputs": [
        [
            5.59353495
        ]
    ]
}
```

## 参见

+   [TensorFlow 文档：模型服务](https://oreil.ly/5ZEQo)

# 23.6 在 Seldon 中为 PyTorch 模型提供服务

## 问题

您希望为实时预测提供经过训练的 PyTorch 模型。

## 解决方案

使用 Seldon Core Python 包装器提供模型服务：

```py
# Import libraries
import torch
import torch.nn as nn
import logging

# Create a PyTorch model class
class SimpleNeuralNet(nn.Module):
    def __init__(self):
        super(SimpleNeuralNet, self).__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(10, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16,16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
            torch.nn.Dropout(0.1), # Drop 10% of neurons
            torch.nn.Sigmoid(),

        )
# Create a Seldon model object with the name `MyModel`
class MyModel(object):

    # Loads the model
    def __init__(self):
        self.network = SimpleNeuralNet()
        self.network.load_state_dict(
            torch.load("model.pt")["model_state_dict"],
            strict=False
        )
        logging.info(self.network.eval())

    # Makes a prediction
    def predict(self, X, features_names=None):
        return self.network.forward(X)
```

并使用 Docker 运行它：

```py
docker run -it -v $(pwd):/app -p 9000:9000 kylegallatin/seldon-example
    seldon-core-microservice MyModel --service-type MODEL
```

```py
2023-03-11 14:40:52,277 - seldon_core.microservice:main:578 -
    INFO:  Starting microservice.py:main
2023-03-11 14:40:52,277 - seldon_core.microservice:main:579 -
    INFO:  Seldon Core version: 1.15.0
2023-03-11 14:40:52,279 - seldon_core.microservice:main:602 -
    INFO:  Parse JAEGER_EXTRA_TAGS []
2023-03-11 14:40:52,287 - seldon_core.microservice:main:605 -
    INFO:  Annotations: {}
2023-03-11 14:40:52,287 - seldon_core.microservice:main:609 -
    INFO:  Importing MyModel
2023-03-11 14:40:55,901 - root:__init__:25 - INFO:  SimpleNeuralNet(
  (sequential): Sequential(
    (0): Linear(in_features=10, out_features=16, bias=True)
    (1): ReLU()
    (2): Linear(in_features=16, out_features=16, bias=True)
    (3): ReLU()
    (4): Linear(in_features=16, out_features=1, bias=True)
    (5): Dropout(p=0.1, inplace=False)
    (6): Sigmoid()
  )
)
2023-03-11 14:40:56,024 - seldon_core.microservice:main:640 -
    INFO:  REST gunicorn microservice running on port 9000
2023-03-11 14:40:56,028 - seldon_core.microservice:main:655 -
    INFO:  REST metrics microservice running on port 6000
2023-03-11 14:40:56,029 - seldon_core.microservice:main:665 -
    INFO:  Starting servers
2023-03-11 14:40:56,029 - seldon_core.microservice:start_servers:80 -
    INFO:  Using standard multiprocessing library
2023-03-11 14:40:56,049 - seldon_core.microservice:server:432 -
    INFO:  Gunicorn Config:  {'bind': '0.0.0.0:9000', 'accesslog': None,
    'loglevel': 'info', 'timeout': 5000, 'threads': 1, 'workers': 1,
    'max_requests': 0, 'max_requests_jitter': 0, 'post_worker_init':
    <function post_worker_init at 0x7f5aee2c89d0>, 'worker_exit':
    functools.partial(<function worker_exit at 0x7f5aee2ca170>,
    seldon_metrics=<seldon_core.metrics.SeldonMetrics object at
    0x7f5a769f0b20>), 'keepalive': 2}
2023-03-11 14:40:56,055 - seldon_core.microservice:server:504 -
    INFO:  GRPC Server Binding to 0.0.0.0:5000 with 1 processes.
2023-03-11 14:40:56,090 - seldon_core.wrapper:_set_flask_app_configs:225 -
    INFO:  App Config:  <Config {'ENV': 'production', 'DEBUG': False,
    'TESTING': False, 'PROPAGATE_EXCEPTIONS': None, 'SECRET_KEY': None,
    'PERMANENT_SESSION_LIFETIME': datetime.timedelta(days=31),
    'USE_X_SENDFILE': False, 'SERVER_NAME': None, 'APPLICATION_ROOT': '/',
    'SESSION_COOKIE_NAME': 'session', 'SESSION_COOKIE_DOMAIN': None,
    'SESSION_COOKIE_PATH': None, 'SESSION_COOKIE_HTTPONLY': True,
    'SESSION_COOKIE_SECURE': False, 'SESSION_COOKIE_SAMESITE': None,
    'SESSION_REFRESH_EACH_REQUEST': True, 'MAX_CONTENT_LENGTH': None,
    'SEND_FILE_MAX_AGE_DEFAULT': None, 'TRAP_BAD_REQUEST_ERRORS': None,
    'TRAP_HTTP_EXCEPTIONS': False, 'EXPLAIN_TEMPLATE_LOADING': False,
    'PREFERRED_URL_SCHEME': 'http', 'JSON_AS_ASCII': None,
    'JSON_SORT_KEYS': None, 'JSONIFY_PRETTYPRINT_REGULAR': None,
    'JSONIFY_MIMETYPE': None, 'TEMPLATES_AUTO_RELOAD': None,
    'MAX_COOKIE_SIZE': 4093}>
2023-03-11 14:40:56,091 - seldon_core.wrapper:_set_flask_app_configs:225 -
    INFO:  App Config:  <Config {'ENV': 'production', 'DEBUG': False,
    'TESTING': False, 'PROPAGATE_EXCEPTIONS': None, 'SECRET_KEY': None,
    'PERMANENT_SESSION_LIFETIME': datetime.timedelta(days=31),
    'USE_X_SENDFILE': False, 'SERVER_NAME': None, 'APPLICATION_ROOT': '/',
    'SESSION_COOKIE_NAME': 'session', 'SESSION_COOKIE_DOMAIN': None,
    'SESSION_COOKIE_PATH': None, 'SESSION_COOKIE_HTTPONLY': True,
    'SESSION_COOKIE_SECURE': False, 'SESSION_COOKIE_SAMESITE': None,
    'SESSION_REFRESH_EACH_REQUEST': True, 'MAX_CONTENT_LENGTH': None,
    'SEND_FILE_MAX_AGE_DEFAULT': None, 'TRAP_BAD_REQUEST_ERRORS': None,
    'TRAP_HTTP_EXCEPTIONS': False, 'EXPLAIN_TEMPLATE_LOADING': False,
    'PREFERRED_URL_SCHEME': 'http', 'JSON_AS_ASCII': None,
    'JSON_SORT_KEYS': None, 'JSONIFY_PRETTYPRINT_REGULAR': None,
    'JSONIFY_MIMETYPE': None, 'TEMPLATES_AUTO_RELOAD': None,
    'MAX_COOKIE_SIZE': 4093}>
2023-03-11 14:40:56,096 - seldon_core.microservice:_run_grpc_server:466 - INFO:
    Starting new GRPC server with 1 threads.
[2023-03-11 14:40:56 +0000] [23] [INFO] Starting gunicorn 20.1.0
[2023-03-11 14:40:56 +0000] [23] [INFO] Listening at: http://0.0.0.0:6000 (23)
[2023-03-11 14:40:56 +0000] [23] [INFO] Using worker: sync
[2023-03-11 14:40:56 +0000] [30] [INFO] Booting worker with pid: 30
[2023-03-11 14:40:56 +0000] [1] [INFO] Starting gunicorn 20.1.0
[2023-03-11 14:40:56 +0000] [1] [INFO] Listening at: http://0.0.0.0:9000 (1)
[2023-03-11 14:40:56 +0000] [1] [INFO] Using worker: sync
[2023-03-11 14:40:56 +0000] [34] [INFO] Booting worker with pid: 34
2023-03-11 14:40:56,217 - seldon_core.gunicorn_utils:load:103 - INFO:
    Tracing not active
```

## 讨论

虽然我们可以使用多种方式为 PyTorch 模型提供服务，但在这里我们选择了 Seldon Core Python 包装器。Seldon Core 是一个流行的用于在生产环境中为模型提供服务的框架，具有许多有用的功能，使其比 Flask 应用程序更易于使用和更可扩展。它允许我们编写一个简单的类（上面我们使用 `MyModel`），而 Python 库则负责所有服务器组件和端点。然后我们可以使用 `seldon-core-microservice` 命令运行服务，该命令启动一个 REST 服务器、gRPC 服务器，甚至公开一个指标端点。要向服务进行预测，我们可以在端口 9000 上调用以下端点：

```py
curl -X POST http://127.0.0.1:9000/predict  -H 'Content-Type: application/json'
    -d '{"data": {"ndarray":[[0, 0, 0, 0, 0, 0, 0, 0, 0]]}}'
```

您应该看到以下输出：

```py
{"data":{"names":["t:0","t:1","t:2","t:3","t:4","t:5","t:6","t:7","t:8"],
    "ndarray":[[0,0,0,0,0,0,0,0,0]]},"meta":{}}
```

## 参见

+   [Seldon Core Python 包](https://oreil.ly/FTofY)

+   [TorchServe 文档](https://oreil.ly/fjmrE)
