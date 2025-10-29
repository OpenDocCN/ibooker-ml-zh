# 第十二章：在引擎盖和更远的地方

在本章中，我们将涉及一些我们在前几章中用于模拟的方法。

我们已经概述了这个要点：在基于仿真的代理学习中，代理经历一个*训练*过程，为其行为开发一个*策略*。策略充当了从先前的*观察*到其响应的*动作*及相应*奖励*的映射。训练发生在大量的*回合*中，期间累积奖励应随着代理在给定任务中的改进而增加，部分受*超参数*的控制，这些超参数控制训练期间代理行为的各个方面，包括生成行为模型的*算法*。

一旦训练完成，*推理*被用来查询训练代理模型以响应给定刺激（观察）的适当行为（动作），但学习已经停止，因此代理不再会在给定任务上*改进*。

我们已经讨论了大部分这些概念：

+   我们了解观察、动作和奖励以及它们之间映射如何用于建立策略。

+   我们知道训练阶段会在大量的回合中进行，一旦完成，代理就会转向推理（仅查询模型，不再更新它）。

+   我们知道我们将超参数文件传递给`mlagents -learn`过程，但我们在这部分有点草率。

+   我们知道在训练过程中有不同的算法可供选择，但也许并不清楚为什么会选择其中的特定选项。

因此，本节将进一步探讨 ML-Agents 中可用的*超参数*和*算法*，以及何时以及为何选择使用它们中的每一个，以及它们如何影响您在训练中的其他选择，如奖励方案的选择。

# 超参数（和参数）

当使用 ML-Agents 开始训练时，需要传递一个 YAML 文件，其中包含必要的*超参数*。这在机器学习世界通常被称为*超参数文件*，但其中不仅包含这些，因此 ML-Agents 文档更倾向于将其称为*配置文件*。其中包含在学习过程中使用的变量，其值将执行以下操作之一：

+   指定训练过程的各个方面（“参数”）

+   更改代理或模型本身在学习过程中的行为（“超参数”）

## 参数

常见配置的训练*参数*包括以下内容：

`trainer_type`

训练中使用的算法，从`ppo`、`sac`或`poca`中选择

`max_steps`

在一个回合结束之前，代理可以接收的最大观察次数或执行的动作数量，无论是否已实现目标

`checkpoint_interval`和`keep_checkpoints`

训练过程中输出重复/备份模型的频率以及保留的最新模型数量

`summary_freq`

如何定期输出（或发送到 TensorBoard）有关训练进展的详细信息。

`network_settings`及其对应的子参数

允许您指定代表代理策略的神经网络的一些大小、形状或行为。

选择`trainer_type`将取决于您的代理和环境的各个方面。我们将在接下来的章节中深入讨论这些算法的内部工作原理。

定义`max_steps`很重要，因为如果模拟运行了数千次或数百万次——如在模型训练过程中发生的那样——很可能在某个时刻代理会陷入无法恢复的状态。如果没有强制限制，代理将保持在这种状态，并持续污染其行为模型，使用不相关或不具代表性的数据，直到 Unity 耗尽可用内存或用户终止进程。这不是理想的情况。相反，应使用经验或估计来确定一个数字，允许一个缓慢前进的代理实现目标，但不允许过度挣扎。

例如，假设您正在训练一辆自动驾驶车辆，需要穿过一个赛道并达到目标位置，先前训练过的模型或测试运行告诉您，理想的运行需要大约 5,000 步才能到达目标位置。如果将`max_steps`设置为`500,000`，那么一个*无用*的情节，即代理未能达成任何成就，将导致信息量增加十倍，这很可能会混淆模型的过程。但是，如果将`max_steps`设置为`5,000`或更低，模型将永远没有机会取得中等结果，而是每次都会在很短的时间内结束尝试，直到最终（也许）偶然达到了完美的无先验知识情节。这种可能性极小。在这些数字之间选择最佳的策略；例如，在这个例子中大约为`10,000`步。对于您自己的代理，理想值将取决于其执行任务的复杂性。

检查点功能允许在每次`checkpoint_interval`更改时保存中间模型，始终保留最后`keep_checkpoints`个模型。这意味着，如果您发现代理在训练结束前表现良好，您可以终止训练过程，仅使用最新的检查点。还可以从检查点恢复训练，允许从一种算法开始训练，然后切换到另一种——例如在第一章中提到的从 BC 到 GAIL 的示例。

访问`network_settings`可用于设置决定行为模型神经网络形状的子参数。如果要在存储或计算资源有限的环境（例如边缘设备）中使用生成的模型，则这对于减小模型大小非常有帮助，例如通过`network_settings→num_layers`或`network_settings->hidden_units`。其他设置可以更精细地控制特定方面，如用于解释视觉输入的方法，或者是否对连续观测进行归一化。

## 奖励参数

接下来需要定义进一步的参数来指定如何在训练期间处理奖励。这些参数属于`reward_signals`。

在使用明确奖励时，必须设置`reward_signals->extrinsic->strength`和`extrinsic->gamma`。在这里，`strength`只是一个应用于通过`AddReward`或`SetReward`调用发送的奖励的比例，如果您正在尝试混合学习方法或其他外在和内在奖励的组合，则可能希望减少这一比例。

同时，`gamma`是应用于奖励估计的比例，基于达到该奖励所需的时间。当代理考虑下一步该做什么时会使用这一值，基于它认为对每个选项将会收到的奖励。`gamma`可以被视为一个衡量代理应该在长期获益与短期获益之间优先考虑多少的指标。代理是否应该放弃未来几步的奖励，以希望在最后实现更大的目标并获得更大的奖励？还是应该立即做出能立即获得奖励的选择？这个选择将取决于您的奖励方案以及代理要完成任务的复杂性，但通常较高的值（表明更多的长期思考）倾向于产生更智能的代理。

还可以启用其他类型的奖励，这些奖励直接来自训练过程，并被称为*内在*奖励——类似于模仿学习方法使用的那些奖励。但是，IL 奖励代理与展示行为的相似性，而这些奖励鼓励了一般属性，几乎像某种特定代理的*个性*。

最普遍适用的内在奖励是`curiosity`，由`reward_signals->curiosity`定义。代理的好奇心意味着优先级别映射未知（即尝试新事物并查看其得分）超过已知良好动作（即过去给他们带来得分的动作）。通过基于动作的新颖程度或其结果的意外性的奖励来鼓励好奇心，这有助于避免稀疏奖励环境中常见的局部最大化问题。

例如，一个代理可能被设计成寻找并站在一个平台上以打开一个门，然后穿过打开的门到达目标。为了激励每一步并加快训练速度，你可以给予代理站在平台上的奖励，并对达到目标后给予指数级增长的奖励。但一个不好奇的代理可能会意识到它在第一步获得了奖励，并决定最佳行动是不断地在平台上站立直至每一集结束，最终结束训练。这是因为它*知道*站在平台上是好的，而它认为其他任何行动的结果（在它看来可能是无限的）是未知的。因此，它将坚持自己擅长的行为。这就是为什么多阶段目标通常需要引入人工好奇心，使代理更愿意尝试新事物。

要启用好奇心，只需传递与外在奖励所需的相同的超参数：

`reward_signals->curiosity->strength`

当试图平衡好奇奖励与其他奖励（如外在奖励）时，需要按比例缩放好奇奖励的数字（必须在`0.0`和`1.0`之间）。

`reward_signals->curiosity->gamma`

第二个尺度用于根据实现所需时间来调节奖励的感知价值，与`extrinsic->gamma`中的相同（同样在`0.0`和`1.0`之间）。

其他不太常用的内在奖励信号可以用来引入其他代理倾向，例如随机网络蒸馏，或者启用特定的学习类型，例如 GAIL。

## 超参数

常见配置的模型*超参数*包括：

`batch_size`

控制模型每次更新时模拟执行的迭代次数。如果你完全使用连续动作，`batch_size`应该很大（数千），如果只使用离散动作，则应该很小（十几）。

`buffer_size`

取决于算法的不同，控制不同的事物。对于 PPO 和 MA-POCA，它控制在更新模型之前收集的经验数量（应该是`batch_size`的几倍）。对于 SAC，`buffer_size`对应于经验缓冲区的最大大小，因此 SAC 可以从旧的和新的经验中学习。

`learning_rate`

对模型的影响每次更新的程度。通常介于`1e-5`和`1e-3`之间。

###### 提示

如果训练不稳定（换句话说，奖励不一致增加），尝试降低`learning_rate`。较大的`buffer_size`也将对训练的稳定性有所帮助。

一些超参数是特定于所使用的训练器的。让我们从您可能经常使用的训练器（PPO）的重要参数开始：

+   `beta`鼓励探索。在这里，较高的`beta`值将导致与好奇心内在奖励类似的结果，因为它鼓励代理尝试新事物，即使早期已发现了奖励。在简单环境中更倾向于较低的`beta`值，因此代理将倾向于限制其行为在过去曾有益的行为上，这可能会减少训练时间。

+   `epsilon`决定了行为模型对变化的接受程度。在这里，较高的`epsilon`值将允许代理快速采纳新行为一旦发现奖励，但这也意味着代理的行为容易改变，甚至在训练后期也是如此。较低的`epsilon`值意味着代理需要更多尝试才能从经验中学习，可能会导致更长的训练时间，但会确保在训练后期行为一致。

+   调度超参数如`beta_schedule`和`epsilon_schedule`可用于在训练过程中更改其他超参数的值，例如在早期训练中优先考虑`beta`（好奇心），或在后期减少`epsilon`（善变性）。

###### 小贴士

POCA/MA-POCA 使用与 PPO 相同的超参数，但是[SAC 有一些专门的超参数](https://oreil.ly/SuApG)。

欲知当前 ML-Agents 支持的参数和超参数的完整列表，请参阅[ML-Agents 文档](https://oreil.ly/4M3An)。

###### 小贴士

如果你不太了解（或者不想了解）你选择的模型所需的超参数是什么意思，你可以查看 GitHub 上每种训练器类型的[ML-Agents 示例文件](https://oreil.ly/0048b)。一旦你看到训练的过程，如果你遇到像这里描述的一些特定于超参数的问题，你可以选择调整这些具体的数值。

# 算法

Unity ML-Agents 框架允许代理通过优化定义在以下方式之一的奖励来学习行为：

明确地（外在奖励）由你

在我们使用`AddReward`或`SetReward`方法的 RL 方法中：当我们知道代理做对了某事时（或者做错了时）我们给予奖励（或惩罚）。

隐含地（内在奖励）由选择的算法

在 IL 方法中，基于与提供的行为演示的相似性来给予奖励：我们向代理展示行为，它试图克隆它，并根据其克隆的质量自动获得奖励。

隐含地由训练过程

我们讨论的超参数设置，其中代理可以因展现某些属性（如好奇心）而受到奖励。在这本书中，我们并没有深入讨论这个，因为这超出了本书的范围。

但这并不是 ML-Agents 中可用算法之间的唯一区别。

*近端策略优化（PPO）* 可能是在使用 Unity 进行 ML 工作时最明智的默认选择。[PPO](https://oreil.ly/6rCeP) 试图逼近一个理想函数，将代理的观察映射到给定状态下可能的最佳行动。它被设计为通用算法。它可能不是最有效的，但通常可以完成任务。这就是为什么 Unity ML-Agents 将其作为默认选项的原因。

*软策演员-评论者（SAC）* 是一种*离策略*的强化学习算法。这基本上意味着可以单独定义最佳训练行为和最佳结果代理行为。这可以减少代理达到最佳行为所需的训练时间，因为可以在训练期间鼓励一些可能在训练中可取但在最终代理行为模型中不可取的特征。

这种属性的最佳例子是*好奇心*。在训练期间，好奇心的探索是很好的，因为你不希望你的代理只发现一个给它点数的东西，然后再也不尝试其他任何事物。但是一旦模型训练完毕，这种探索就不那么理想了，因为如果训练如期进行，它已经发现了所有理想的行为。

因此，[SAC](https://oreil.ly/jPlxp) 在训练速度上可能更快，但相比于像 PPO 这样的*在策略*方法，需要更多的内存来存储和更新单独的行为模型。

###### 注意

有关于 PPO 是*在策略*还是*离策略*的争论。我们倾向于将其视为在策略上，因为它基于遵循当前策略进行更新。

*多代理人死后信用分配（POCA 或 MA-POCA）* 是一种多代理算法，使用集中的*评论者*来奖励和惩罚一组代理。奖励类似于基本的 PPO，但是奖励给评论者。代理应该学会如何最好地贡献以获得奖励，但也可以单独奖励。它被认为是*死后*的，因为代理在学习过程中可以从代理组中移除，但仍然会学习其行为对组获得奖励的贡献，即使在被移除后也是如此。这意味着代理可以采取对组有益的行动，即使这些行动会导致它们自身的死亡。

我们在第九章中使用了[MA-POCA](https://oreil.ly/aDvvz)。

# Unity 推理引擎和集成

在代理训练期间，代表代理行为的神经网络会随着代理执行动作并接收奖励形式的反馈而不断更新。这通常是一个漫长的过程，因为神经网络图可能非常庞大，调整所需的计算量会随其大小而增加。同样，使代理在所需任务中持续成功所需的剧集数量通常在数十万甚至数百万。

因此，在 ML-Agents 中训练一个中等复杂度的代理可能会占用个人电脑数小时甚至数天。然而，一个*训练过的*代理可以轻松地包含在 Unity 游戏中或导出用于简单的应用程序中。那么，他们在训练后的使用如何变得更加可行呢？

###### 提示

如果您想要在 ML-Agents 外部训练用于 ML-Agents 的模型，请首先了解[Tensor 名称](https://oreil.ly/J59hl)和[Barracuda 模型参数](https://oreil.ly/pu3YM)。这超出了本书的范围，但非常有趣！

答案在于训练期间所需的性能与*推理*期间的差异。在训练阶段后，代理行为的神经网络被锁定在一个位置；随着代理执行动作，它将不再更新，奖励也不再作为反馈发送。相反，将以与训练期间相同的方式向代理提供相同的观察，但是定义哪些观察将与哪些动作响应的规则已经定义好了。

弄清楚对应的反应就像追踪一个图形一样简单。因此，推理是一个高效的过程，即使在计算资源有限的应用程序中也可以包含。所需的只是一个*推理引擎*，它知道如何接受输入，追踪网络图，并输出适当的操作。

###### 注意

Unity ML-Agents 推理引擎是使用*计算着色器*实现的，这些着色器是在 GPU 上运行的小型专用程序（也称为图形卡），但不用于图形处理。这意味着它们可能无法在所有[平台](https://oreil.ly/Iaj5s)上工作。

幸运的是，Unity ML-Agents 自带一个称为*Unity 推理引擎*（有时称为 Barracuda）的推理引擎。因此，你不需要制作自己的引擎，或者在训练时使用的底层框架（如 PyTorch 或 TensorFlow）。

###### 提示

您可以在[Unity 文档](https://oreil.ly/0jyye)中了解更多关于 Barracuda 的信息。

如果你在 ML-Agents 外部训练了一个模型，你将无法使用它与 Unity 的推理引擎。理论上，你可以创建一个符合 ML-Agents 期望的常量和张量名称的模型，但这并没有得到官方支持。

###### 警告

机器学习代理生成的模型，使用 CPU 运行推理速度可能比使用 GPU 更快，这可能有点违反直觉（或者对于你的背景来说可能很直观），除非你的代理有大量的视觉观察。

## 使用 ML-Agents Gym 包装器

OpenAI Gym 是一个（几乎成为事实上的标准）用于开发和探索强化学习算法的开源库。在本节中，我们将快速了解使用 ML-Agents Gym 包装器来探索强化学习算法。

在开始使用 ML-Agents Gym Wrapper 之前，您需要设置好 Python 和 Unity 环境。因此，如果您还没有这样做，请通过“设置”的步骤创建一个新的环境。完成这些步骤后，继续执行以下操作：

1.  激活您的新环境，然后安装`gym_unity` Python 包：

    ```
    pip install gym_unity
    ```

1.  然后，您可以从任何 Python 脚本中启动 Unity 仿真环境*作为 gym*：

    ```
    from gym_unity.envs import UnityToGymWrapper
    env = UnityToGymWrapper
        (unity_env, uint8_visual, flatten_branched, allow_multiple_obs)
    ```

在这种情况下，`unity_env`是要包装并作为 gym 呈现的 Unity 环境。就是这样！

### Unity 环境和 OpenAI Baselines

OpenAI 项目中最有趣的组成部分之一是 OpenAI Baselines，这是一组高质量的强化学习算法实现。现在处于维护模式，但它仍然提供了一系列非常有用的算法，供您探索强化学习。

方便的是，您可以通过 Unity ML-Agents Gym Wrapper 与 Unity 仿真环境一起使用 OpenAI Baselines。

作为一个快速示例，我们将使用 OpenAI 的 DQN 算法来训练我们在第十一章中使用的 GridWorld。

首先，您需要构建 GridWorld 环境的一个副本：

1.  在您克隆或下载的 ML-Agents GitHub 存储库的副本中，打开项目文件夹（参见“设置”），作为 Unity 项目使用 Unity Hub 打开，并使用项目视图打开 GridWorld 场景。

1.  然后打开“文件”菜单 → “构建设置”，选择您当前的平台。

1.  确保“场景构建列表”中仅选中了 GridWorld 场景。

1.  点击“构建”并选择一个您熟悉的位置保存构建。

    ###### 提示

    您可能会想知道为什么我们不能使用默认的注册表来获取 GridWorld 的副本，因为我们在这里专门使用 Python。原因是 ML-Agents Gym Wrapper 只支持存在单个代理的环境。所有预构建的默认注册表环境都有多个区域，以加快训练速度。

    接下来，我们将转到 Python：

1.  在您的 Python 环境中，您需要安装 Baselines 包：

    ```
    pip install git+git://github.com/openai/baselines
    ```

    ###### 警告

    您可能需要在执行此操作之前安装 TensorFlow，通过`pip install tensorflow==1.15`。您将需要这个特定版本的 TensorFlow 以保持与 OpenAI Baselines 的兼容性：特别是它使用了 TensorFlow 的`contrib`模块，该模块不是 TensorFlow 2.0 的一部分。这就是 Python 的乐趣所在。

1.  接下来，启动 Jupyter Lab，按照我们在“尝试环境”中使用的过程创建一个新的笔记本。

1.  添加以下`import`行：

    ```
    import gym

    from baselines import deepq
    from baselines import logger

    from mlagents_envs.environment import UnityEnvironment
    from gym_unity.envs import UnityToGymWrapper
    ```

1.  接下来，获取一下我们刚刚构建的 Unity 环境，并将其转换为 gym 环境：

    ```
    unity_env =
        UnityEnvironment("/Users/parisba/Downloads/GridWorld.app", 10000, 1)
    env = UnityToGymWrapper(unity_env, uint8_visual=True)
    logger.configure('./logs') # Change to log in a different directory
    ```

    注意，`/Users/parisba/Downloads/GridWorld.app`的等效部分应该指向一个*.app*或*.exe*或其他可执行文件（取决于您的平台），这是我们刚刚制作的 GridWorld 的构建副本。

1.  最后，运行训练：

    ```
    act = deepq.learn(
        env,
        "cnn", # For visual inputs
        lr=2.5e-4,
        total_timesteps=1000000,
        buffer_size=50000,
        exploration_fraction=0.05,
        exploration_final_eps=0.1,
        print_freq=20,
        train_freq=5,
        learning_starts=20000,
        target_network_update_freq=50,
        gamma=0.99,
        prioritized_replay=False,
        checkpoint_freq=1000,
        checkpoint_path='./logs', # Save directory
        dueling=True
    )
    print("Saving model to unity_model.pkl")
    act.save("unity_model.pkl")
    ```

你的环境将启动，并将使用 OpenAI Baselines DQN 算法进行训练。

## Side Channels

Unity 的 Python ML-Agents 组件提供了一个名为“side channels”的功能，允许你在运行在 Unity 中的 C#代码和 Python 代码之间双向共享任意信息。具体来说，ML-Agents 提供了两个可供使用的 side channels：`EngineConfigurationChannel`和`EnvironmentParametersChannel`。

### 引擎配置通道

引擎配置通道允许你变化与引擎相关的参数：时间尺度、图形质量、分辨率等。它旨在通过变化质量来提高训练性能，或者在推断期间使事物更漂亮、更有趣或更有用以供人类审查。

按照以下步骤创建一个`EngineConfigurationChannel`：

1.  确保以下内容包含在你的`import`语句中：

    ```
    from mlagents_envs.environment import UnityEnvironment
    from mlagents_envs.side_channel.engine_configuration_channel
        import EngineConfigurationChannel
    ```

1.  创建一个`EngineConfigurationChannel`：

    ```
    channel = EngineConfigurationChannel()
    ```

1.  将通道传递给你正在使用的`UnityEnvironment`：

    ```
    env = UnityEnvironment(side_channels=[channel])
    ```

1.  根据需要配置通道：

    ```
    channel.set_configuration_parameters(time_scale = 2.0)
    ```

在这种情况下，这个`EngineConfigurationChannel`的配置将`time_scale`设置为`2.0`。

就这样！有一系列可能用于`set_configuration_parameters`的参数，比如用于分辨率控制的`width`和`height`，`quality_level`和`target_frame_rate`。

### 环境参数通道

环境参数通道比引擎配置通道更通用；它允许你在 Python 和仿真环境之间传递任何需要的数值值。

按照以下步骤创建一个`EnvironmentParametersChannel`：

1.  确保你拥有以下的`import`语句：

    ```
    from mlagents_envs.environment import UnityEnvironment
    from mlagents_envs.side_channel.environment_parameters_channel import
        EnvironmentParametersChannel
    ```

1.  创建一个`EnvironmentParametersChannel`并将其传递给`UnityEnvironment`，就像我们对引擎配置通道所做的那样：

    ```
    channel = EnvironmentParametersChannel()
    env = UnityEnvironment(side_channels=[channel])
    ```

1.  接下来，在 Python 端使用该通道，命名一个参数为`set_float_parameter`：

    ```
    channel.set_float_parameter("myParam", 11.0)
    ```

    在这种情况下，参数被命名为`myParam`。

1.  这允许你从 Unity 中的 C#访问相同的参数：

    ```
    var environment_parameters = Academy.Instance.EnvironmentParameters;
    float myParameterValue = envParameters.GetWithDefault("myParam", 0.0f);
    ```

这里调用中的`0.0f`是一个默认值。

到此为止，我们完成了本章的内容，也基本完成了书籍中的模拟。在代码下载中提供了一些下一步操作；如果你对强化学习感兴趣并希望进一步探索，请打开资源包中书籍网站的 Next_Steps 文件夹。
