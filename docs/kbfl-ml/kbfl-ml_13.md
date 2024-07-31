# 附录 B. 特定于云的工具和配置

特定于云的工具可以加速您的开发，但也可能导致供应商锁定。

# Google Cloud

由于 Kubeflow 起源于 Google，因此在 Google Cloud 上运行时会有一些额外的功能可用。我们将快速指出如何使用 TPUs 和 Dataflow 加速您的机器学习流水线，以及更多与 Google 相关的组件可以在 [Kubeflow GitHub 仓库](https://oreil.ly/F7c9l) 中找到。

## TPU 加速实例

机器学习过程的不同部分不仅可以从不同数量的机器中受益，还可以从不同类型的机器中受益。最常见的例子是模型服务：通常，大量低内存的机器可以表现得相当不错，但对于模型训练，高内存或 TPU 加速的机器可以提供更大的好处。虽然使用 GPU 有一个方便的内置缩写，但对于 TPU，您需要显式地 `import kfp.gcp as gcp`。一旦导入了 kfp 的 gcp，您可以通过在容器操作中添加 `.apply(gcp.use_tpu(tpu_cores=cores, tpu_resource=version, tf_version=tf_version))` 的方式，向任何容器操作添加 TPU 资源。

###### 警告

TPU 节点仅在特定地区可用。请查看 [此 Google Cloud 页面](https://oreil.ly/1HAzM) 以获取支持的地区列表。

## Dataflow 用于 TFX

在 Google Cloud 上，您可以配置 Kubeflow 的 TFX 组件使用 Google 的 Dataflow 进行分布式处理。为此，您需要指定一个分布式输出位置（因为工作节点之间没有共享持久卷），并配置 TFX 使用 Dataflow 运行器。展示这一点最简单的方法是重新查看 示例 5-8; 要使用 Dataflow，我们会将其更改为 示例 B-1。

##### 示例 B-1\. 将管道更改为使用 Dataflow

```
generated_output_uri = root_output_uri + kfp.dsl.EXECUTION_ID_PLACEHOLDER
beam_pipeline_args = [
    '--runner=DataflowRunner',
    '--project=' + project_id,
    '--temp_location=' + root_output_uri + '/tmp'),
    '--region=' + gcp_region,
    '--disk_size_gb=50', # Adjust as needed
]

records_example = tfx_csv_gen(
    input_uri=fetch.output, # Must be on distributed storage
    beam_pipeline_args=beam_pipeline_args,
    output_examples_uri=generated_output_uri)
```

正如您所见，将管道更改为使用 Dataflow 相对简单，并且可以打开更大规模的数据进行处理。

虽然特定于云的加速可以带来好处，但要小心权衡，看看是否值得为了未来可能需要更换提供商而带来的额外麻烦。
