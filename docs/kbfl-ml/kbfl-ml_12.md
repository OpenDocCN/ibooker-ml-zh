# 附录 A\. Argo 执行器配置和权衡

直到最近，所有 Kubernetes 实现都支持 Docker API。最初的 Argo 实现依赖于它们。随着 [OpenShift 4](https://oreil.ly/bIoqk) 的推出，不再支持 Docker API，情况发生了变化。为了支持没有 Docker API 的情况，Argo 引入了几个新的执行器：Docker、Kubelet 和 Kubernetes API。在 Argo 参数文件中，`containerRuntimeExecutor` 配置值控制使用哪个执行器。根据这里的信息，每个执行器的优缺点总结在 表 A-1 中。这张表应该帮助您选择正确的 Argo 执行器值。

表 A-1\. Argo 和 Kubernetes API

| 执行器 | Docker | Kubelet | Kubernetes API | PNC |
| --- | --- | --- | --- | --- |
| 优点 | 支持所有工作流示例。最可靠、经过充分测试，非常可扩展。与 Docker 守护程序进行沟通，用于处理重型任务。 | 安全性高。无法绕过 Pod 的服务账号权限。中等可扩展性。日志检索和容器轮询针对 Kubelet 执行。 | 安全性高。无法绕过 Pod 的服务账号权限。无需额外配置。 | 安全性高。无法绕过服务账号权限。可以从基础镜像层收集构件。可扩展性强：进程轮询通过 procfs 完成，而非 kubelet/k8s API。 |
| 缺点 | 安全性最低。需要挂载主机的 `docker.sock`（通常被 OPA 拒绝）。 | 可能需要额外的 kubelet 配置。只能在卷（例如 `emptyDir`）中保存参数/构件，而不能保存基础镜像层（例如 `/tmp`）。 | 可扩展性最低。日志检索和容器轮询针对 k8s API 服务器执行。只能在卷（例如 `emptyDir`）中保存参数/构件，而不能保存基础镜像层（例如 `/tmp`）。 | 进程不再以 pid 1 运行。对于完成过快的容器，可能会导致构件收集失败。无法从挂载在其下的卷中捕获基础镜像层的构件目录。尚未成熟。 |
| Argo 配置 | docker | kubelet | k8sapi | pns |
