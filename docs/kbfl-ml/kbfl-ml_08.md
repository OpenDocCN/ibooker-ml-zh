# 第七章：训练一个机器学习模型

在 第五章 中，我们学习了如何准备和清理我们的数据，这是机器学习流程中的第一步。现在让我们深入探讨如何利用我们的数据来训练一个机器学习模型。

训练通常被认为是机器学习中的“大部分”工作。我们的目标是创建一个函数（即“模型”），能够准确预测它之前没有见过的结果。直观地说，模型训练非常像人类学习新技能的方式——我们观察、练习、纠正错误，并逐渐改进。在机器学习中，我们从一个可能不太擅长其工作的初始模型开始。然后，我们将模型通过一系列的训练步骤，将训练数据馈送给模型。在每个训练步骤中，我们将模型产生的预测结果与真实结果进行比较，并查看模型的表现如何。然后，我们调整这个模型的参数（例如，通过改变每个特征所赋予的权重），试图提高模型的准确性。一个好的模型是能够在不过度拟合特定输入集的情况下进行准确预测的模型。

在本章中，我们将学习如何使用两种不同的库——TensorFlow 和 Scikit-learn 来训练机器学习模型。TensorFlow 在 Kubeflow 中有原生的一流支持，而 Scikit-learn 则没有。但正如我们在本章中所看到的，这两个库都可以很容易地集成到 Kubeflow 中。我们将演示如何在 Kubeflow 的笔记本中对模型进行实验，以及如何将这些模型部署到生产环境中。

# 使用 TensorFlow 构建推荐系统

让我们首先来了解 TensorFlow——这是一个由 Google 开发的机器学习开源框架。它目前是实现深度学习的最流行库之一，特别是在机器学习驱动的应用程序中。TensorFlow 对于包括 CPU、GPU 和 TPU 在内的各种硬件的计算任务有很好的支持。我们选择 TensorFlow 进行这个教程是因为它的高级 API 用户友好，并且抽象了许多复杂的细节。

让我们通过一个简单的教程来熟悉 TensorFlow。在 第一章 中，我们介绍了我们的案例研究之一，即面向客户的产品推荐系统。在本章中，我们将使用 TensorFlow 来实现这个系统。具体来说，我们将做两件事情：

1.  使用 TensorFlow 来训练一个产品推荐模型。

1.  使用 Kubeflow 将训练代码封装并部署到生产集群中。

TensorFlow 的高级 Keras API 使得实现我们的模型相对容易。事实上，大部分模型可以用不到 50 行 Python 代码来实现。

###### 提示

Keras 是用于深度学习模型的高级 TensorFlow API。它具有用户友好的界面和高可扩展性。此外，Keras 还预置了许多常见的神经网络实现，因此您可以立即启动一个模型。

让我们首先选择我们的推荐系统模型。我们从一个简单的假设开始——如果两个人（Alice 和 Bob）对一组产品有相似的意见，那么他们在其他产品上的看法也更可能相似。换句话说，Alice 更有可能与 Bob 拥有相同的偏好，而不是随机选择的第三个人。因此，我们可以仅使用用户的购买历史来构建推荐模型。这就是协同过滤的理念——我们从许多用户（因此称为“协同”）那里收集偏好信息，并使用这些数据进行选择性预测（因此称为“过滤”）。

要构建这个推荐模型，我们需要几件事：

用户的购买历史

我们将使用来自[此 GitHub 仓库的示例输入数据](https://oreil.ly/F-8rS)。

数据存储

为了确保我们的模型能够跨不同的平台工作，我们将使用 MinIO 作为存储系统。

训练模型

我们使用的实现基于[Github 上的 Keras 模型](https://oreil.ly/hTGQf)。

我们将首先使用 Kubeflow 的笔记本服务器对该模型进行实验，然后使用 Kubeflow 的 TFJob API 将训练工作部署到我们的集群上。

## 入门

让我们从下载先决条件开始。您可以从[本书的 GitHub 仓库](https://oreil.ly/Kubeflow_for_ML_ch07)下载笔记本。要运行笔记本，您需要一个运行中包含 MinIO 服务的 Kubeflow 集群。请查看“支持组件”来配置 MinIO。确保还安装了 MinIO 客户端（“mc”）。

我们还需要准备数据以便进行训练：您可以从[这个 GitHub 站点](https://oreil.ly/BK6XS)下载用户购买历史数据。然后，您可以使用 MinIO 客户端创建存储对象，如示例 7-1 中所示。

##### 示例 7-1. 设置先决条件

```
# Port-forward the MinIO service to http://localhost:9000
kubectl port-forward -n kubeflow svc/minio-service 9000:9000 &

# Configure MinIO host
mc config host add minio http://localhost:9000 minio minio123

# Create storage bucket
mc mb minio/data

# Copy storage objects
mc cp go/src/github.com/medium/items-recommender/data/recommend_1.csv \\
        minio/data/recommender/users.csv
mc cp go/src/github.com/medium/items-recommender/data/trx_data.csv \\
        minio/data/recommender/transactions.csv
```

## 启动新的笔记本会话

现在让我们通过创建一个新的笔记本开始。您可以通过在 Kubeflow 仪表板的“笔记本服务器”面板中导航，然后点击“New Server”并按照说明操作来完成此操作。对于本示例，我们使用`tensorFlow-1.15.2-notebook-cpu:1.0`镜像。¹

当笔记本服务器启动时，请点击右上角的“上传”按钮并上传`Recommender_Kubeflow.ipynb`文件。单击文件以启动新会话。

代码的前几部分涉及导入库并从 MinIO 中读取训练数据。然后，我们对输入数据进行归一化处理，以便准备开始训练。这个过程称为特征准备，我们在第五章中讨论过。在本章中，我们将专注于练习的模型训练部分。

## TensorFlow 训练

现在我们的笔记本已设置并准备好数据，我们可以创建一个 TensorFlow 会话，如示例 7-2 所示。²

##### 示例 7-2\. 创建 TensorFlow 会话

```
# Create TF session and set it in Keras
sess = tf.Session()
K.set_session(sess)
K.set_learning_phase(1)
```

对于模型类，我们使用协作过滤的示例 7-3 中的代码。

##### 示例 7-3\. 深度协同过滤学习

```
class DeepCollaborativeFiltering(Model):
   def__init__(self, n_customers, n_products, n_factors, p_dropout = 0.2):
      x1 = Input(shape = (1,), name="user")

      P = Embedding(n_customers, n_factors, input_length = 1)(x1)
      P = Reshape((n_factors,))(P)

      x2 = Input(shape = (1,), name="product")

      Q = Embedding(n_products, n_factors, input_length = 1)(x2)
      Q = Reshape((n_factors,))(Q)

      x = concatenate([P, Q], axis=1)
      x = Dropout(p_dropout)(x)

      x = Dense(n_factors)(x)
      x = Activation('relu')(x)
      x = Dropout(p_dropout)(x)

      output = Dense(1)(x)

      super(DeepCollaborativeFiltering, self).__init__([x1, x2], output)

   def rate(self, customer_idxs, product_idxs):
      if (type(customer_idxs) == int and type(product_idxs) == int):
          return self.predict([np.array(customer_idxs).reshape((1,)),\
                  np.array(product_idxs).reshape((1,))])

      if (type(customer_idxs) == str and type(product_idxs) == str):
          return self.predict( \
                 [np.array(customerMapping[customer_idxs]).reshape((1,)),\
                 np.array(productMapping[product_idxs]).reshape((1,))])

      return self.predict([
         np.array([customerMapping[customer_idx] \
                for customer_idx in customer_idxs]),
            np.array([productMapping[product_idx] \
                for product_idx in product_idxs])
      ])
```

这是我们模型类的基础。它包括一个构造函数，其中包含一些用于使用 Keras API 实例化协作过滤模型的代码，以及一个“rate”函数，我们可以使用我们的模型进行预测——即客户对特定产品的评分。

我们可以像示例 7-4 那样创建一个模型实例。

##### 示例 7-4\. 模型创建

```
model = DeepCollaborativeFiltering(n_customers, n_products, n_factors)
model.summary()
```

现在我们准备开始训练我们的模型。我们可以通过设置一些超参数来实现这一点，如示例 7-5 所示。

##### 示例 7-5\. 设置训练配置

```
bs = 64
val_per = 0.25
epochs = 3
```

这些是控制训练过程的超参数。它们通常在训练开始之前设置，不像模型参数那样是从训练过程中学习的。设置正确的超参数值可以显著影响模型的有效性。目前，让我们为它们设置一些默认值。在第十章中，我们将学习如何使用 Kubeflow 调整超参数。

现在我们已准备好运行训练代码。查看示例 7-6。

##### 示例 7-6\. 拟合模型

```
model.compile(optimizer = 'adam', loss = mean_squared_logarithmic_error)
model.fit(x = [customer_idxs, product_idxs], y = ratings,
        batch_size = bs, epochs = epochs, validation_split = val_per)
print('Done training!')
```

训练完成后，你应该能看到类似于示例 7-7 中的结果。

##### 示例 7-7\. 模型训练结果

```
Train on 100188 samples, validate on 33397 samples
Epoch 1/3
100188/100188 [==============================]
- 21s 212us/step - loss: 0.0105 - val_loss: 0.0186
Epoch 2/3
100188/100188 [==============================]
- 20s 203us/step - loss: 0.0092 - val_loss: 0.0188
Epoch 3/3
100188/100188 [==============================]
- 21s 212us/step - loss: 0.0078 - val_loss: 0.0192
Done training!
```

恭喜你：你已成功在 Jupyter 笔记本中训练了一个 TensorFlow 模型。但我们还没有完成——为了以后能够使用我们的模型，我们应该先导出它。你可以通过设置使用 MinIO Client 的导出目的地来完成此操作，如示例 7-8 所示。

##### 示例 7-8\. 设置导出目的地

```
directorystream = minioClient.get_object('data', 'recommender/directory.txt')
directory = ""
for d in directorystream.stream(32*1024):
    directory += d.decode('utf-8')
arg_version = "1"
export_path = 's3://models/' + directory + '/' + arg_version + '/'
print ('Exporting trained model to', export_path)
```

一旦设置了导出目的地，你可以像示例 7-9 那样导出模型。

##### 示例 7-9\. 导出模型

```
# Inputs/outputs
tensor_info_users = tf.saved_model.utils.build_tensor_info(model.input[0])
tensor_info_products = tf.saved_model.utils.build_tensor_info(model.input[1])
tensor_info_pred = tf.saved_model.utils.build_tensor_info(model.output)

print ("tensor_info_users", tensor_info_users.name)
print ("tensor_info_products", tensor_info_products.name)
print ("tensor_info_pred", tensor_info_pred.name)

# Signature
prediction_signature = (tf.saved_model.signature_def_utils.build_signature_def(
        inputs={"users": tensor_info_users, "products": tensor_info_products},
        outputs={"predictions": tensor_info_pred},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
# Export
legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
builder = tf.saved_model.builder.SavedModelBuilder(export_path)
builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
          prediction_signature,
      },
      legacy_init_op=legacy_init_op)
builder.save()
```

现在我们准备使用这个模型来提供预测，正如我们将在第八章中学到的那样。但在此之前，让我们看看如何使用 Kubeflow 部署这个训练作业。

# 部署 TensorFlow 训练任务

到目前为止，我们已经使用 Jupyter 笔记本进行了一些 TensorFlow 训练，这是原型设计和实验的好方法。但很快我们可能会发现我们的原型不足——也许我们需要使用更多数据来完善模型，或者我们需要使用专用硬件来训练模型。有时候，我们甚至可能需要持续运行训练作业，因为我们的模型在不断发展。或许更重要的是，我们的模型必须能够部署到生产环境中，以便为实际的客户请求提供服务。

为了处理这些需求，我们的训练代码必须易于打包和部署到不同的环境中。实现这一点的一种方法是使用 TFJob——一个 Kubernetes 自定义资源（使用 Kubernetes 操作者 `tf-operator` 实现），您可以使用它在 Kubernetes 上运行 TensorFlow 训练作业。

我们将首先将我们的推荐器部署为单容器 TFJob。由于我们已经有一个 Python 笔记本，将其导出为 Python 文件非常简单——只需选择“文件”，然后选择“另存为”，然后选择“Python”。这将保存您的笔记本作为一个可以立即执行的 Python 文件。

接下来的步骤是将训练代码打包到容器中。可以通过 Dockerfile 完成，就像在示例 7-10 中看到的那样。

##### 示例 7-10\. TFJob Dockerfile

```
FROM tensorflow/tensorflow:1.15.2-py3
RUN pip3 install --upgrade pip
RUN pip3 install pandas --upgrade
RUN pip3 install keras --upgrade
RUN pip3 install minio --upgrade
RUN mkdir -p /opt/kubeflow
COPY Recommender_Kubeflow.py /opt/kubeflow/
ENTRYPOINT ["python3", "/opt/kubeflow/Recommender_Kubeflow.py"]
```

接下来，我们需要将此容器及其所需的库一起构建，并将容器映像推送到存储库：

```
docker build -t kubeflow/recommenderjob:1.0 .
docker push kubeflow/recommenderjob:1.0
```

完成后，我们准备创建 TFJob 的规范，就像在示例 7-11 中所示。

##### 示例 7-11\. 单容器 TFJob 示例

```
apiVersion: "kubeflow.org/v1"   ![1](img/1.png)
kind: "TFJob"                   ![2](img/2.png)
metadata:
  name: "recommenderjob"        ![3](img/3.png)
spec:
  tfReplicaSpecs:               ![4](img/4.png)
    Worker:
      replicas: 1
    restartPolicy: Never
    template:
      spec:
        containers:
        - name: tensorflow image: kubeflow/recommenderjob:1.0
```

![1](img/#co_training_a_machine_learning_model_CO1-1)

`apiVersion`字段指定您正在使用的 TFJob 自定义资源的版本。需要在您的 Kubeflow 集群中安装相应的版本（在本例中为 v1）。

![2](img/#co_training_a_machine_learning_model_CO1-2)

`kind`字段标识自定义资源的类型——在本例中是 TFJob。

![3](img/#co_training_a_machine_learning_model_CO1-3)

`metadata`字段适用于所有 Kubernetes 对象，并用于在集群中唯一标识对象——您可以在此处添加名称、命名空间和标签等字段。

![4](img/#co_training_a_machine_learning_model_CO1-4)

架构中最重要的部分是`tfReplicaSpecs`。这是您的 TensorFlow 训练集群及其期望状态的实际描述。在此示例中，我们只有一个工作节点副本。在接下来的部分中，我们将进一步检查这个字段。

您的 TFJob 还有一些其他可选配置，包括：

`activeDeadlineSeconds`

在系统可以终止作业之前保持作业活动的时间。如果设置了此项，则系统将在截止日期到期后终止作业。

`backoffLimit`

将作业标记为失败之前重试此作业的次数。例如，将其设置为 3 意味着如果作业失败 3 次，系统将停止重试。

`cleanPodPolicy`

配置是否在作业完成后清理 Kubernetes pods。设置此策略可以用于保留用于调试的 pods。可以设置为 All（清理所有 pods）、Running（仅清理运行中的 pods）或 None（不清理 pods）。

现在将 TFJob 部署到您的集群中，就像示例 7-12 中所示。

##### 示例 7-12\. 部署 TFJob

```
kubectl apply -f recommenderjob.yaml
```

您可以使用以下命令监视 TFJob 的状态，如示例 7-13。

##### 示例 7-13\. 查看 TFJob 的状态

```
kubectl describe tfjob recommenderjob
```

这应该显示类似于示例 7-14 的内容。

##### 示例 7-14\. TF 推荐作业描述

```
Status:
  Completion Time:  2019-05-18T00:58:27Z
  Conditions:
    Last Transition Time:  2019-05-18T02:34:24Z
    Last Update Time:      2019-05-18T02:34:24Z
    Message:               TFJob recommenderjob is created.
    Reason:                TFJobCreated
    Status:                True
    Type:                  Created
    Last Transition Time:  2019-05-18T02:38:28Z
    Last Update Time:      2019-05-18T02:38:28Z
    Message:               TFJob recommenderjob is running.
    Reason:                TFJobRunning
    Status:                False
    Type:                  Running
    Last Transition Time:  2019-05-18T02:38:29Z
    Last Update Time:      2019-05-18T02:38:29Z
    Message:               TFJob recommenderjob successfully completed.
    Reason:                TFJobSucceeded
    Status:                True
    Type:                  Succeeded
  Replica Statuses:
    Worker:
      Succeeded:  1
```

注意，状态字段包含一系列作业条件，表示作业转换到每个状态的时间。这对调试非常有用——如果作业失败，作业失败的原因将在此处显示。

到目前为止，我们使用了一些训练样本数量适中的相对简单和直接的模型进行了训练。在实际生活中，学习更复杂的模型可能需要大量训练样本或模型参数。这样的模型可能过大且计算复杂，无法由单台机器处理。这就是分布式训练发挥作用的地方。

# 分布式训练

到目前为止，我们已经在 Kubeflow 上部署了一个单工作节点的 TensorFlow 作业。它被称为“单工作节点”，因为从托管数据到执行实际训练步骤的所有工作都在单台机器上完成。然而，随着模型变得更复杂，单台机器通常不够用——我们可能需要将模型或训练样本分布到多台联网机器上。TensorFlow 支持分布式训练模式，其中训练在多个工作节点上并行执行。

分布式训练通常有两种方式：数据并行和模型并行。在数据并行中，训练数据被分成多个块，每个块上运行相同的训练代码。在每个训练步骤结束时，每个工作节点向所有其他节点通信其更新。模型并行则相反——所有工作节点使用相同的训练数据，但模型本身被分割。在每个步骤结束时，每个工作节点负责同步模型的共享部分。

TFJob 接口支持多工作节点分布式训练。在概念上，TFJob 是与训练作业相关的所有资源的逻辑分组，包括*pods*和*services*。在 Kubeflow 中，每个复制的工作节点或参数服务器都计划在自己的单容器 pod 上。为了使副本能够相互同步，每个副本都需要通过端点暴露自己，这是一个 Kubernetes 内部服务。将这些资源在父资源（即 TFJob）下逻辑地分组允许这些资源一起协同调度和垃圾回收。

在本节中，我们将部署一个简单的 MNIST 示例，并进行分布式训练。TensorFlow 训练代码已为您提供，可以在 [此 GitHub 仓库](https://oreil.ly/ySztV) 中找到。

让我们看一下分布式 TensorFlow 作业的 YAML 文件，位于 示例 7-15 中。

##### 示例 7-15\. 分布式 TFJob 示例

```
apiVersion: "kubeflow.org/v1"
kind: "TFJob"
metadata:
  name: "mnist"
  namespace: kubeflow
spec:
  cleanPodPolicy: None
  tfReplicaSpecs:
    Worker:
      replicas: 2
      restartPolicy: Never
      template:
        spec:
          containers:
            - name: tensorflow
              image: gcr.io/kubeflow-ci/tf-mnist-with-summaries:1.0
              command:
                - "python"
                - "/var/tf_mnist/mnist_with_summaries.py"
                - "--log_dir=/train/logs"
                - "--learning_rate=0.01"
                - "--batch_size=150"
              volumeMounts:
                - mountPath: "/train"
                  name: "training"
          volumes:
            - name: "training"
              persistentVolumeClaim:
                claimName: "tfevent-volume"
```

注意，`tfReplicaSpecs` 字段现在包含几种不同的副本类型。在典型的 TensorFlow 训练集群中，有几种可能的情况：

主节点

负责编排计算任务、发出事件并对模型进行检查点处理

参数服务器

为模型参数提供分布式数据存储

工作节点

这是实际进行计算和训练的地方。当主节点未明确定义（如前述示例中），其中一个工作节点充当主节点。

评估者

评估者可以用于在训练模型时计算评估指标。

还要注意，副本规范包含多个属性，描述其期望的状态：

`replicas`

应为此副本类型生成多少副本

`template`

描述每个副本要创建的 pod 的 `PodTemplateSpec`

`restartPolicy`

确定 pod 退出时是否重新启动。允许的值如下：

`Always`

意味着 pod 将始终重新启动。这个策略适用于参数服务器，因为它们永不退出，应在失败事件中始终重新启动。

`OnFailure`

意味着如果 pod 由于失败而退出，将重新启动 pod。非零退出代码表示失败。退出代码为 0 表示成功，pod 将不会重新启动。这个策略适用于主节点和工作节点。

`ExitCode`

意味着重新启动行为取决于 TensorFlow 容器的退出代码如下：

+   0 表示进程成功完成，将不会重新启动。

+   1–127 表示永久错误，容器将不会重新启动。

+   128–255 表示可重试错误，容器将被重新启动。这个策略适用于主节点和工作节点。

`Never`

意味着终止的 pod 将永远不会重新启动。这个策略应该很少使用，因为 Kubernetes 可能会因各种原因（如节点变得不健康）而终止 pod，并且这会阻止作业恢复。

编写完 TFJob 规范后，将其部署到您的 Kubeflow 集群：

```
kubectl apply -f dist-mnist.yaml
```

监控作业状态类似于单容器作业：

```
kubectl describe tfjob mnist
```

这应该输出类似于 示例 7-16 的内容。

##### 示例 7-16\. TFJob 执行结果

```
Status:
  Completion Time:  2019-05-12T00:58:27Z
  Conditions:
    Last Transition Time:  2019-05-12T00:57:31Z
    Last Update Time:      2019-05-12T00:57:31Z
    Message:               TFJob dist-mnist-example is created.
    Reason:                TFJobCreated
    Status:                True
    Type:                  Created
    Last Transition Time:  2019-05-12T00:58:21Z
    Last Update Time:      2019-05-12T00:58:21Z
    Message:               TFJob dist-mnist-example is running.
    Reason:                TFJobRunning
    Status:                False
    Type:                  Running
    Last Transition Time:  2019-05-12T00:58:27Z
    Last Update Time:      2019-05-12T00:58:27Z
    Message:               TFJob dist-mnist-example successfully completed.
    Reason:                TFJobSucceeded
    Status:                True
    Type:                  Succeeded
  Replica Statuses:
    Worker:
      Succeeded:  2
```

注意，`Replica Statuses` 字段显示了每个副本类型的状态细分。当所有工作节点都完成时，TFJob 将成功完成。如果有任何工作节点失败，则 TFJob 的状态也将失败。

## 使用 GPU

GPU 是由许多较小且专业的核心组成的处理器。最初设计用于渲染图形，GPU 越来越多地用于大规模并行计算任务，如机器学习。与 CPU 不同，GPU 非常适合在其多个核心上分发大型工作负载并同时执行。

要使用 GPU 进行训练，您的 Kubeflow 集群需要预先配置以启用 GPU。请参考您的云服务提供商的文档以启用 GPU 使用。在集群上启用 GPU 后，您可以通过修改命令行参数在训练规范中的特定副本类型上启用 GPU，例如示例 7-17。

##### 示例 7-17\. TFJob with GPU example

```
    Worker:
      replicas: 4
      restartPolicy: Never
      template:
        spec:
          containers:
            - name: tensorflow
              image: kubeflow/tf-dist-mnist-test:1.0
              args:
            - python
            - /var/tf_dist_mnist/dist_mnist.py
            - --num_gpus=1
```

## 使用其他框架进行分布式训练

Kubeflow 被设计为一个多框架的机器学习平台。这意味着分布式训练的模式可以轻松扩展到其他框架。截至本文撰写时，已编写了许多运算符，为其他框架（包括 PyTorch 和 Caffe2）提供一流支持。

示例 7-18 展示了 PyTorch 训练作业规范的样子。

##### 示例 7-18\. Pytorch Distributed Training Example

```
apiVersion: "kubeflow.org/v1"
kind: "PyTorchJob"
metadata:
  name: "pytorch-dist"
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: gcr.io/kubeflow-ci/pytorch-dist-sendrecv-test:1.0
    Worker:
      replicas: 3
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: gcr.io/kubeflow-ci/pytorch-dist-sendrecv-test:1.0
```

如您所见，其格式与 TFJobs 非常相似。唯一的区别在于副本类型。

# 使用 Scikit-Learn 训练模型

到目前为止，我们已经看到如何使用 Kubeflow 中的内置运算符来训练机器学习模型。然而，还有许多框架和库没有 Kubeflow 运算符。在这些情况下，您仍然可以在 Jupyter 笔记本³或自定义 Docker 镜像中使用您喜欢的框架。

Scikit-learn 是一个基于 NumPy 构建的用于机器学习的开源 Python 库，用于高性能线性代数和数组操作。该项目起源于 scikits.learn，由 David Cournapeau 在 Google Summer of Code 项目中开发。其名称源于它是一个“SciKit”（SciPy 工具包），是一个单独开发和分发的第三方扩展到 SciPy。Scikit-learn 是 GitHub 上最受欢迎的机器学习库之一，也是维护最好的之一。在 Kubeflow 中，使用 Scikit-learn 训练模型作为通用的 Python 代码支持，无需专门的分布式训练运算符。

该库支持最先进的算法，如 KNN、XGBoost、随机森林和 SVM。Scikit-learn 在 Kaggle 竞赛和知名技术公司中被广泛使用。Scikit-learn 在预处理、降维（参数选择）、分类、回归、聚类和模型选择方面提供帮助。

在本节中，我们将通过使用 Scikit-learn 在[1994 年美国人口普查数据集](https://oreil.ly/9nQrt)上训练模型来探索 Kubeflow。该示例基于用于收入预测的 Anchor 解释的[这个实现](https://oreil.ly/9hnha)，并利用了从 1994 年人口普查数据集中提取的数据。数据集包括多个分类变量和连续特征，包括年龄、教育、婚姻状况、职业、工资、关系、种族、性别、原籍国家以及资本收益和损失。我们将使用随机森林算法——一种集成学习方法，用于分类、回归以及其他任务，在训练时通过构建大量决策树并输出类的众数（分类）或个体树的平均预测（回归）来运行。

你可以从[本书的 GitHub 存储库](https://oreil.ly/Kubeflow_for_ML_ch07_notebook)下载这个笔记本。

## 开始一个新的笔记本会话

让我们从创建一个新的笔记本开始。类似于 TensorFlow 训练，你可以通过导航到你的 Kubeflow 仪表板中的“笔记本服务器”面板，然后点击“新建服务器”，按照指示操作。例如，我们可以使用`tensorFlow-1.15.2-notebook-cpu:1.0 image`。

###### 提示

在 Kubeflow 中工作时，利用 GPU 资源加速你的 Scikit 模型的一种简单方法是切换到 GPU 类型。

当笔记本服务器启动时，在右上角点击“上传”按钮并上传*IncomePrediction.ipynb*文件。点击文件以启动一个新的会话。

## 数据准备

笔记本的前几个部分涉及导入库和读取数据。然后我们继续进行特征准备。⁴ 为了特征转换，我们使用 Scikit-learn 的管道。管道使得将一致数据输入模型变得更加容易。

对于我们的随机森林训练，我们需要定义有序（标准化数据）和分类（独热编码）特征，就像示例 7-19 中那样。

##### 示例 7-19\. 特征准备

```
ordinal_features = [x for x in range(len(feature_names))
                if x not in list(category_map.keys())]
ordinal_transformer = Pipeline(steps=[
    ('imputer',  SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = list(category_map.keys())
categorical_transformer = Pipeline(steps=[('imputer',
    SimpleImputer(strategy='median')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
```

###### 提示

许多真实世界的数据集包含缺失值，这些值被数据特定的占位符（如空白和 NaN）编码。这种数据集通常与假设所有值为数值的 Scikit-learn 估计器不兼容。有多种策略可以处理这种缺失数据。一个基本的策略是丢弃包含缺失值的整行和/或列，这样做的代价是丢失数据。一个更好的策略是填补缺失值——通过从已知数据部分推断出它们。`Simple imputer`是一个 Scikit-learn 类，允许你通过用指定的预定义值替换 NaN 值来处理预测模型数据集中的缺失数据。

一旦定义了特征，我们可以使用列转换器将它们组合，如示例 7-20 所示。

##### 示例 7-20\. 使用列转换器组合列

```
preprocessor = ColumnTransformer(transformers=[
    ('num', ordinal_transformer, ordinal_features),
    ('cat', categorical_transformer, categorical_features)])
preprocessor.fit(X_train)
```

###### 提示

Scikit-learn 的独热编码用于将分类特征编码为独热数字数组。编码器将整数或字符串数组转换，用分类（离散）特征替换值。使用独热编码方案对特征进行编码，为每个类别创建一个二进制列，并返回一个稀疏矩阵或密集数组（取决于 sparse 参数）。

转换器本身看起来像示例 7-21。

##### 示例 7-21\. 数据变换器

```
ColumnTransformer(n_jobs=None, remainder='drop', sparse_threshold=0.3,
  transformer_weights=None,
  transformers=[('num',
    Pipeline(memory=None,
      steps=[
        ('imputer', SimpleImputer(add_indicator=False,
          copy=True,
          fill_value=None,
          missing_values=nan,
          strategy='median',
          verbose=0)),
        ('scaler', StandardScaler(copy=True,
          with_mean=True,
          with_std=True))],
        verbose=False),
      [0, 8, 9, 10]),
    ('cat',
     Pipeline(memory=None,
       steps=[('imputer', SimpleImputer(add_indicator=False,
         copy=True,
         fill_value=None,
         missing_values=nan,
         strategy='median',
         verbose=0)),
       ('onehot', OneHotEncoder(categories='auto',
         drop=None,
         dtype=<class 'numpy.float64'>,
         handle_unknown='ignore',
         sparse=True))],
       verbose=False),
       [1, 2, 3, 4, 5, 6, 7, 11])],
    verbose=False)
```

由于这种转换，我们的数据以特征的形式准备好进行训练。

## Scikit-Learn 训练

一旦我们准备好特征，就可以开始训练。这里我们将使用由 Scikit-learn 库提供的`RandomForestClassifier`，如示例 7-22 所示。

##### 示例 7-22\. 使用 RandomForestClassifier

```
np.random.seed(0)
clf = RandomForestClassifier(n_estimators=50)
clf.fit(preprocessor.transform(X_train), Y_train)
```

###### 提示

机器学习算法的集合和特定特征是选择特定机器学习实现框架的主要驱动因素之一。即使在不同框架中相同算法的实现提供略有不同的特性，这些特性可能（或可能不）对您的特定数据集很重要。

一旦预测完成，我们可以评估训练结果，如示例 7-23 所示。

##### 示例 7-23\. 评估训练结果

```
predict_fn = lambda x: clf.predict(preprocessor.transform(x))
print('Train accuracy: ', accuracy_score(Y_train, predict_fn(X_train)))
print('Test accuracy: ', accuracy_score(Y_test, predict_fn(X_test)))
```

该部分将返回示例 7-24 中的结果。

##### 示例 7-24\. 训练结果

```
Train accuracy:  0.9655333333333334
Test accuracy:  0.855859375
```

在这一点上，模型已创建并可以通过导出直接使用（参见下一节）。模型的一个最重要的属性是其可解释性。虽然模型的可解释性主要用于模型服务，但对于模型创建也很重要，有两个主要原因：

+   如果在模型创建期间模型服务中的可解释性很重要，我们经常需要验证所创建的模型是否可解释。

+   许多模型解释方法要求在模型创建过程中进行额外的计算。

基于此，我们将展示如何在模型创建过程中实现模型可解释性⁵。

## 解释模型

对于模型解释，我们使用锚点，这是[Seldon's Alibi project](https://oreil.ly/VSGxe)的一部分。

该算法提供了适用于应用于图像、文本和表格数据的分类模型的模型无关（黑盒）和人类可解释的解释。连续特征被离散化为分位数（例如，分位数），因此它们变得更易解释。在候选锚点中，保持特征不变（相同类别或分箱为离散化特征），同时从训练集中对其他特征进行抽样，如示例 7-25 所示。

##### 示例 7-25\. 定义表格锚点

```
explainer = AnchorTabular(
    predict_fn, feature_names, categorical_names=category_map, seed=1)
explainer.fit(X_train, disc_perc=[25, 50, 75])
```

这创建了表格锚点（示例 7-26）。

##### 示例 7-26\. 表格锚点

```
AnchorTabular(meta={
    'name': 'AnchorTabular',
    'type': ['blackbox'],
    'explanations': ['local'],
    'params': {'seed': 1, 'disc_perc': [25, 50, 75]}
})
```

现在我们可以为测试集中第一条观测的预测获取一个锚点。*锚点*是一个充分条件——即，当锚点成立时，预测应与示例 7-27 中此实例的预测相同。

##### 示例 7-27\. 预测计算

```
idx = 0
class_names = adult.target_names
print('Prediction: ', class_names[explainer.predictor( \
                X_test[idx].reshape(1, -1))[0]])
```

返回如示例 7-28 所示的预测计算结果。

##### 示例 7-28\. 预测计算结果

```
Prediction:  <=50K
```

我们将精度阈值设置为`0.95`。这意味着在锚定条件成立的观察中，预测至少 95%的时间将与解释实例的预测相同。现在我们可以为这个预测获取解释（示例 7-29）。

##### 示例 7-29\. 模型解释

```
explanation = explainer.explain(X_test[idx], threshold=0.95)
print('Anchor: %s' % (' AND '.join(explanation.anchor)))
print('Precision: %.2f' % explanation.precision)
print('Coverage: %.2f' % explanation.coverage)
```

返回如示例 7-30 所示的模型解释结果。

##### 示例 7-30\. 模型解释结果

```
Anchor: Marital Status = Separated AND Sex = Female
Precision: 0.95
Coverage: 0.18
```

这告诉我们，决策的主要因素是婚姻状况（`Separated`）和性别（`Female`）。并非所有点都可以找到锚点。让我们尝试为测试集中的另一条观测获取一个锚点——一个预测为`>50K`的示例，如示例 7-31 所示。

##### 示例 7-31\. 模型解释

```
idx = 6
class_names = adult.target_names
print('Prediction: ', class_names[explainer.predictor( \
                X_test[idx].reshape(1, -1))[0]])

explanation = explainer.explain(X_test[idx], threshold=0.95)
print('Anchor: %s' % (' AND '.join(explanation.anchor)))
print('Precision: %.2f' % explanation.precision)
print('Coverage: %.2f' % explanation.coverage)
```

返回如示例 7-32 所示的模型解释结果。

##### 示例 7-32\. 模型解释结果

```
Prediction:  >50K
Could not find a result satisfying the 0.95 precision constraint.
Now returning the best non-eligible result.
Anchor: Capital Loss > 0.00 AND Relationship = Husband AND
    Marital Status = Married AND Age > 37.00 AND
    Race = White AND Country = United-States AND Sex = Male
Precision: 0.71
Coverage: 0.05
```

由于数据集不平衡（大约 25:75 高:低收入者比例），在采样阶段与低收入者对应的特征范围将被过度采样。因此，在这种情况下找不到锚点。这是一个特征，因为它可以指出数据集不平衡，但也可以通过生成平衡数据集来修复，以便为任何类找到锚点。

## 模型导出

为了将创建的模型用于服务，我们需要导出模型。这可以通过 Scikit-learn 功能完成，如示例 7-33 所示。

##### 示例 7-33\. 导出模型

```
dump(clf, '/tmp/job/income.joblib')
```

这将以 Scikit-learn 格式导出一个模型，例如，可以由 Scikit-learn 服务器用于推断。

## 集成到流水线中

无论您想使用哪个基于 Python 的机器学习库，如果 Kubeflow 没有相应的操作符，您可以按照正常方式编写您的代码，然后将其容器化。要将本章中构建的笔记本作为流水线阶段使用，请参阅“将整个笔记本用作数据准备流水线阶段”。在这里，我们可以使用`file_output`将生成的模型上传到我们的工件跟踪系统，但您也可以使用持久卷机制。

# 结论

在本章中，我们看了如何使用两种非常不同的框架（TensorFlow 和 Scikit-learn）在 Kubeflow 中训练机器学习模型。

我们学习了如何使用 TensorFlow 构建协同过滤推荐系统。我们使用 Kubeflow 创建了一个笔记本会话，在那里我们使用了 Keras API 原型化了一个 TensorFlow 模型，然后使用 TFJob API 将我们的训练作业部署到了 Kubernetes 集群。最后，我们看了如何使用 TFJob 进行分布式训练。

我们还学习了如何使用 Scikit-learn 训练通用的 Python 模型，这是 Kubeflow 不原生支持的框架。第九章讨论了如何集成不受支持的非 Python 机器学习系统，这有点复杂。尽管 Kubeflow 的一方面训练操作员可以简化您的工作，但重要的是要记住您并不受此限制。

在第八章中，我们将探讨如何提供在本章中训练的模型。

¹ 当前，Kubeflow 提供了带有 TensorFlow 1.15.2 和 2.1.0 的 CPU 和 GPU 镜像，或者您可以使用自定义镜像。

² 本章中的示例使用了 TensorFlow 1.15.2。您可以在[这个 Kubeflow GitHub 网站](https://oreil.ly/I71lt)找到使用 TensorFlow 2.1.0 的示例。

³ 目前 Jupyter 笔记本支持的语言包括 Python、R、Julia 和 Scala。

⁴ 请参阅第五章详细讨论特征准备。

⁵ 有关模型可解释性的更多信息，请参阅[Rui Aguiar 的这篇博客文章](https://oreil.ly/juWml)。
