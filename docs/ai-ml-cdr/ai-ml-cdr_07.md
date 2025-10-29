# 第五章：自然语言处理简介

自然语言处理（NLP）是人工智能中处理理解人类语言的技术。它涉及编程技术，用于创建能够理解语言、分类内容，甚至生成和创作新人类语言组合的模型。在接下来的几章中，我们将探讨这些技术。还有很多服务使用 NLP 创建应用程序，如聊天机器人，但这不在本书的范围内——相反，我们将研究 NLP 的基础以及如何建模语言，以便你能训练神经网络理解和分类文本。稍作调剂，你还将看到如何利用机器学习模型的预测元素来写一些诗歌！

我们将从如何将语言分解成数字开始这一章节，并探讨这些数字如何在神经网络中使用。

# 将语言编码成数字

你可以用多种方式将语言编码成数字。最常见的方法是按字母编码，就像在程序中存储字符串时自然而然地做的那样。然而，在内存中，你不是存储字母*a*，而是它的编码——也许是 ASCII 或 Unicode 值，或者其他什么。例如，考虑单词*listen*。可以用 ASCII 将其编码为数字 76、73、83、84、69 和 78。这样做很好，因为现在你可以用数字来代表这个词。但是再考虑一下单词*silent*，它是*listen*的反字。同样的数字表示那个单词，尽管顺序不同，这可能会使建立理解文本的模型变得有些困难。

###### 注意

*反字*是一个单词，它是另一个单词的字谜，但意思相反。例如，*united*和*untied*是反字，*restful*和*fluster*也是，*Santa*和*Satan*，*forty-five*和*over fifty*。我的职称过去是开发者福音使，但现在改为开发者倡导者——这是一件好事，因为*福音使*是*邪恶的代理人*的反字！

一个更好的选择可能是使用数字来编码整个单词而不是其中的字母。在这种情况下，*silent*可以是数字*x*，*listen*可以是数字*y*，它们不会彼此重叠。

使用这种技术，考虑一句话像“I love my dog.”。你可以用数字[1, 2, 3, 4]来编码它。如果你想编码“I love my cat.”，它可能是[1, 2, 3, 5]。你已经到了能够告诉这些句子有相似含义的地步，因为它们在数值上相似——[1, 2, 3, 4]看起来很像[1, 2, 3, 5]。

这个过程称为*标记化*，接下来你将学习如何在代码中实现它。

## 开始标记化

TensorFlow Keras 包含一个名为`preprocessing`的库，提供了许多非常有用的工具来准备机器学习数据。其中之一是一个`Tokenizer`，它允许您将单词转换为标记。让我们用一个简单的例子来看它的工作原理：

```
`import` tensorflow `as` tf
`from` tensorflow `import` keras
`from` tensorflow.keras.preprocessing.text `import` `Tokenizer`

sentences = [
    `'``Today is a sunny day``'`,
    `'``Today is a rainy day``'`
]

tokenizer = `Tokenizer`(num_words = `100`)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
`print`(word_index)
```

在这种情况下，我们创建了一个`Tokenizer`对象，并指定它可以标记化的单词数。这将是从单词语料库生成的最大标记数。这里我们的语料库非常小，只包含六个唯一的单词，因此我们将远远低于指定的一百个单词。

一旦我们有了一个分词器，调用`fit_on_texts`将创建分词的单词索引。打印出来将显示语料库中单词的一组键/值对，如下所示：

```
`{`'today'`:` `1``,` 'is'`:` `2``,` 'a'`:` `3``,` 'day'`:` `4``,` 'sunny'`:` `5``,` 'rainy'`:` `6``}`
```

分词器非常灵活。例如，如果我们用另一个包含单词“今天”的句子扩展语料库，但后面加上了问号，结果显示它会智能地过滤掉“今天？”只保留“今天”：

```
sentences = [
    `'``Today is a sunny day``'`,
    `'``Today is a rainy day``'`,
    `'``Is it sunny today?``'`
]

{`'``today``'`: `1`, `'``is``'`: `2`, `'``a``'`: `3`, `'``sunny``'`: `4`, `'``day``'`: `5`, `'``rainy``'`: `6`, `'``it``'`: `7`}
```

这种行为由分词器的`filters`参数控制，默认情况下除了撇号字符之外会删除所有标点符号。因此，例如，“今天是个晴天”将变成一个包含[1, 2, 3, 4, 5]的序列，而“今天是晴天吗？”将变成[2, 7, 4, 1]。一旦您的句子中的单词被分词，下一步就是将您的句子转换为数字列表，其中数字是单词作为键的值。

## 将句子转换为序列

现在您已经看到如何将单词分词成数字，下一步是将句子编码成数字序列。这个分词器有一个叫做`text_to_sequences`的方法——您只需将您的句子列表传递给它，它将返回一个序列列表。因此，例如，如果您像这样修改前面的代码：

```
sentences = [
    `'``Today is a sunny day``'`,
    `'``Today is a rainy day``'`,
    `'``Is it sunny today?``'`
]

tokenizer = `Tokenizer`(num_words = `100`)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

`sequences` `=` `tokenizer``.``texts_to_sequences``(``sentences``)`

`print`(sequences)
```

您将获得表示这三个句子的序列。记住单词索引如下：

```
`{`'today'`:` `1``,` 'is'`:` `2``,` 'a'`:` `3``,` 'sunny'`:` `4``,` 'day'`:` `5``,` 'rainy'`:` `6``,` 'it'`:` `7``}`
```

输出将如下所示：

```
[[`1`, `2`, `3`, `4`, `5`], [`1`, `2`, `3`, `6`, `5`], [`2`, `7`, `4`, `1`]]
```

然后您可以用单词替换数字，您将看到句子是有意义的。

现在考虑一下，如果您在一组数据上训练神经网络会发生什么。典型模式是，您有一组用于训练的数据，您知道它不会覆盖您所有的需求，但您希望它尽可能地覆盖。在自然语言处理的情况下，您可能有成千上万个单词在您的训练数据中，用在许多不同的上下文中，但您不可能在每种可能的上下文中都有每个可能的单词。因此，当您向神经网络展示一些新的、以前未见过的文本，包含以前未见过的单词时，可能会发生什么？您猜对了——它会感到困惑，因为它根本没有这些单词的上下文，结果它给出的任何预测都会受到负面影响。

### 使用了超出词汇表的标记

处理这些情况的一种工具是*超出词汇表*（OOV）标记。这可以帮助你的神经网络理解包含以前未见过文本的数据的上下文。例如，考虑前面的小例子语料库，假设你想处理这样的句子：

```
test_data = [
    `'``Today is a snowy day``'`,
    `'``Will it be rainy tomorrow?``'`
]
```

记住，你并不是将此输入添加到现有文本语料库中（可以将其视为你的训练数据），而是考虑预训练网络可能如何查看此文本。如果使用你已经使用过的单词和现有的分词器对其进行分词，就像这样：

```
test_sequences = tokenizer.texts_to_sequences(test_data)
`print`(word_index)
`print`(test_sequences)
```

你的结果将如下所示：

```
`{`'today'`:` `1``,` 'is'`:` `2``,` 'a'`:` `3``,` 'sunny'`:` `4``,` 'day'`:` `5``,` 'rainy'`:` `6``,` 'it'`:` `7``}`
`[``[``1``,` `2``,` `3``,` `5``]``,` `[``7``,` `6``]``]`
```

所以新的句子，将单词替换回标记后，会变成“today is a day”和“it rainy.”

正如你所看到的，你几乎失去了所有的上下文和含义。在这里可能会有帮助的是一个超出词汇表的标记，你可以在分词器中指定它。你可以通过添加一个称为`oov_token`的参数来实现这一点，就像这样——你可以分配任何你喜欢的字符串，但确保它不是语料库中其他地方出现过的字符串：

```
tokenizer = Tokenizer(num_words = 100, `oov_token``=``"``<OOV>``"`)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

test_sequences = tokenizer.texts_to_sequences(test_data)
print(word_index)
print(test_sequences)
```

你会看到输出稍有改善：

```
`{`'<OOV>'`:` `1``,` 'today'`:` `2``,` 'is'`:` `3``,` 'a'`:` `4``,` 'sunny'`:` `5``,` 'day'`:` `6``,` 'rainy'`:` `7``,` 
  'it'`:` `8``}`

`[``[``2``,` `3``,` `4``,` `1``,` `6``]``,` `[``1``,` `8``,` `1``,` `7``,` `1``]``]`
```

你的标记列表有了一个新项目，“<OOV>”，而你的测试句子保持了它们的长度。反向编码它们现在会得到“today is a <OOV> day”和“<OOV> it <OOV> rainy <OOV>.”

前者更接近原始含义。后者因为大部分词汇不在语料库中，仍然缺乏很多上下文，但这是朝着正确方向迈出的一步。

### 理解填充

在训练神经网络时，通常需要使所有数据具有相同的形状。回想一下前几章，当处理图像训练时，你将图像重新格式化为相同的宽度和高度。处理文本时，你面临相同的问题——一旦对单词进行了分词并将句子转换为序列，它们可能具有不同的长度。为了使它们具有相同的大小和形状，你可以使用*填充*。

要探索填充的功能，让我们在语料库中添加另一个更长的句子：

```
sentences = [
    `'``Today is a sunny day``'`,
    `'``Today is a rainy day``'`,
    `'``Is it sunny today?``'`,
    `'``I really enjoyed walking in the snow today``'`
]
```

当你对它进行序列化时，你会看到你的数字列表长度不同：

```
[
  [`2`, `3`, `4`, `5`, `6`], 
  [`2`, `3`, `4`, `7`, `6`], 
  [`3`, `8`, `5`, `2`], 
  [`9`, `10`, `11`, `12`, `13`, `14`, `15`, `2`]
]
```

（当你打印这些序列时，它们会全部在一行上，但我在这里为了清晰起见将它们分成了不同的行。）

如果你想使它们具有相同的长度，你可以使用`pad_sequences` API。首先，你需要导入它：

```
`from` tensorflow.keras.preprocessing.sequence `import` pad_sequences
```

使用这个 API 非常简单。要将（未填充的）序列转换为填充后的集合，你只需调用`pad_sequences`，像这样：

```
padded = pad_sequences(sequences)

`print`(padded)
```

你将得到一组格式良好的序列。它们也会像这样分开显示在不同的行上：

```
[[ `0`  `0`  `0`  `2`  `3`  `4`  `5`  `6`]
 [ `0`  `0`  `0`  `2`  `3`  `4`  `7`  `6`]
 [ `0`  `0`  `0`  `0`  `3`  `8`  `5`  `2`]
 [ `9` `10` `11` `12` `13` `14` `15`  `2`]]
```

序列将被填充为`0`，这不是我们单词列表中的标记。如果你曾想知道为什么标记列表从 1 开始，而程序员通常从 0 开始计数，现在你知道了！

现在你有了一个经过规则化处理的东西，可以用于训练。但在深入讨论之前，让我们稍微探讨一下这个 API，因为它提供了许多可以用来改进数据的选项。

首先，您可能已经注意到，在较短的句子的情况下，为了使它们与最长句子的形状相同，必须在开头添加相应数量的零。这称为*预填充*，这是默认行为。您可以使用`padding`参数来更改此行为。例如，如果您希望您的序列在末尾填充零，可以使用：

```
padded = pad_sequences(sequences, padding=`'``post``'`)
```

由此输出的结果将是：

```
[[ `2`  `3`  `4`  `5`  `6`  `0`  `0`  `0`]
 [ `2`  `3`  `4`  `7`  `6`  `0`  `0`  `0`]
 [ `3`  `8`  `5`  `2`  `0`  `0`  `0`  `0`]
 [ `9` `10` `11` `12` `13` `14` `15`  `2`]]
```

您可以看到现在单词在填充序列的开头，而`0`字符在末尾。

您可能注意到的下一个默认行为是，所有的句子都被制作成与*最长*句子相同的长度。这是一个合理的默认行为，因为它意味着您不会丢失任何数据。这样做的代价是会有很多填充。但是如果您不希望这样，也许因为您有一个非常长的句子，这意味着在填充的序列中会有太多的填充。为了解决这个问题，您可以在调用`pad_sequences`时使用`maxlen`参数，指定所需的最大长度，例如：

```
padded = pad_sequences(sequences, padding='post', `maxlen=6`)
```

由此输出的结果将是：

```
[[ `2`  `3`  `4`  `5`  `6`  `0`]
 [ `2`  `3`  `4`  `7`  `6`  `0`]
 [ `3`  `8`  `5`  `2`  `0`  `0`]
 [`11` `12` `13` `14` `15`  `2`]]
```

现在，您的填充序列的长度都相同，并且填充不过多。不过，您最长的句子确实丢失了一些单词，并且它们被从句子的开头截断了。如果您不想丢失开头的单词，而是希望它们从句子的*末尾*截断，您可以使用`truncating`参数覆盖默认行为，如下所示：

```
padded = pad_sequences(sequences, padding='post', maxlen=6, `truncating='post'`)
```

现在的结果将显示最长的句子现在是在结尾而不是开头截断：

```
[[ `2`  `3`  `4`  `5`  `6`  `0`]
 [ `2`  `3`  `4`  `7`  `6`  `0`]
 [ `3`  `8`  `5`  `2`  `0`  `0`]
 [ `9` `10` `11` `12` `13` `14`]]
```

###### 注意

TensorFlow 支持使用“ragged”（形状不同的）张量进行训练，这非常适合 NLP 的需求。使用它们比我们在本书中覆盖的内容更为高级，但是一旦您完成了接下来几章提供的 NLP 介绍，您可以探索[文档](https://oreil.ly/I1IJW)以获取更多信息。

# 去除停用词和清理文本

在下一节中，您将看到一些真实世界的数据集，您会发现通常有一些您不希望在数据集中的文本。您可能希望过滤掉所谓的*停用词*，它们太常见且没有任何意义，例如“the”，“and”和“but”。您在文本中可能还会遇到许多 HTML 标签，最好有一种清理方法将它们移除。您可能还希望过滤掉其他诸如粗鲁的词、标点符号或姓名之类的内容。稍后我们将探索一个推特数据集，这些推特通常会包含某人的用户 ID，我们希望将其过滤掉。

尽管每个任务基于您的文本语料库都是不同的，但有三件主要的事情可以通过编程方式清理文本。

第一个是去除 HTML 标签。幸运的是，有一个叫做`BeautifulSoup`的库可以轻松实现这一点。例如，如果您的句子包含 HTML 标签如`<br>`，则可以通过以下代码将其删除：

```
`from` `bs4` `import`  BeautifulSoup
`soup` `=` BeautifulSoup`(``sentence``)`
`sentence` `=` `soup``.``get_text``(``)`
```

删除停用词的常见方法是使用停用词列表预处理您的句子，删除停用词的实例。这里有一个简化的例子：

```
`stopwords` `=` `[`"a"`,` "about"`,` "above"`,` ...  "yours"`,` "yourself"`,` "yourselves"`]`
```

完整的停用词列表可以在本章的一些在线[示例](https://oreil.ly/ObsjT)中找到。

然后，在迭代句子时，您可以使用如下代码从句子中删除停用词：

```
words = sentence.split()
filtered_sentence = `"``"`
`for` word `in` words:
    `if` word `not` `in` stopwords:
        filtered_sentence = filtered_sentence + word + `"` `"`
sentences.append(filtered_sentence)
```

您可能还考虑删除标点符号，这可能会误导停用词移除器。刚刚展示的方法寻找被空格包围的单词，因此紧跟在句号或逗号后面的停用词将不会被发现。

使用 Python `string`库提供的翻译函数轻松解决这个问题。它还提供了一个常量，`string.punctuation`，其中包含一组常见的标点符号，因此，要从单词中删除它们，您可以执行以下操作：

```
import  string
`table` `=` `str``.``maketrans``(``'``'``,` `'``'``,` string`.``punctuation``)`
`words` `=` `sentence``.``split``(``)`
`filtered_sentence` `=` `"``"`
for `word` in `words``:`
 `word` `=` `word``.``translate``(``table``)`
  if `word` not  in `stopwords``:`
 `filtered_sentence` `=` `filtered_sentence` `+` `word` `+` `"` `"`
`sentences``.``append``(``filtered_sentence``)`
```

在这里，在过滤停用词之前，句子中的每个单词都删除了标点符号。因此，如果分割句子后得到单词“it;”，它将被转换为“it”，然后作为停用词删除。然而，请注意，当进行此操作时，您可能需要更新停用词列表。这些列表通常包含缩写词和像“you’ll”这样的缩略词。翻译程序将“you’ll”更改为“youll”，如果您希望将其过滤掉，则需要更新您的停用词列表以包含它。

遵循这三个步骤将会为您提供一个更加干净的文本集合。当然，每个数据集都会有其特殊性，您需要与之配合工作。

# 处理真实数据源

现在，您已经了解了获取句子、使用单词索引对其进行编码并排序结果的基本知识，可以通过将一些知名的公共数据集与 Python 提供的工具结合使用，将其转换为可以轻松排序的格式。我们将从 TensorFlow 数据集中已经为您完成了大部分工作的 IMDb 数据集开始。之后，我们将更加亲手操作，处理基于 JSON 的数据集和包含情绪数据的几个逗号分隔值（CSV）数据集！

## 从 TensorFlow 数据集获取文本

我们在第四章中探讨了 TFDS，所以如果您在本节的某些概念上遇到困难，可以快速查看那里。TFDS 的目标是尽可能地简化以标准化方式获取数据的过程。它提供了对多个基于文本的数据集的访问；我们将探索`imdb_reviews`，这是来自互联网电影数据库（IMDb）的 50,000 条带有正面或负面情感标签的影评数据集。

此代码将从 IMDb 数据集加载训练拆分，并遍历它，将包含评论的文本字段添加到名为`imdb_sentences`的列表中。评论是包含评论情感的文本和标签的元组。请注意，通过将`tfds.load`调用包装在`tfds.as_numpy`中，您确保数据将被加载为字符串，而不是张量：

```
imdb_sentences = []
train_data = tfds.as_numpy(tfds.load(`'``imdb_reviews``'`, split=`"``train``"`))
`for` item `in` train_data:
    imdb_sentences.append(str(item[`'``text``'`]))
```

一旦你有了句子，你就可以创建一个分词器并像以前那样将它们拟合到其中，同时创建一个序列集：

```
tokenizer = tf.keras.preprocessing.text.`Tokenizer`(num_words=`5000`)
tokenizer.fit_on_texts(imdb_sentences)
sequences = tokenizer.texts_to_sequences(imdb_sentences)
```

你也可以打印出你的单词索引来检查它：

```
`print`(tokenizer.word_index)
```

它太大了，不能显示整个索引，但这里是前 20 个单词。请注意，分词器按照数据集中单词的频率顺序列出它们，因此像“the”、“and”和“a”这样的常见单词被索引了：

```
{'the': 1, 'and': 2, 'a': 3, 'of': 4, 'to': 5, 'is': 6, 'br': 7, 'in': 8, 
 'it': 9, 'i': 10, 'this': 11, 'that': 12, 'was': 13, 'as': 14, 'for': 15,  
 'with': 16, 'movie': 17, 'but': 18, 'film': 19, "'s": 20, ...}
```

这些是停用词，如前一节所述。由于它们是最常见的单词且不明显，它们可能会影响您的训练准确性。

还要注意，“br”包含在此列表中，因为它在语料库中常用作`<br>`HTML 标签。

你可以更新代码，使用`BeautifulSoup`去除 HTML 标签，添加字符串翻译以删除标点符号，并从给定列表中删除停用词如下：

```
from bs4 import BeautifulSoup
import string

stopwords = ["a", ... , "yourselves"]

table = str.maketrans('', '', string.punctuation)

imdb_sentences = []
train_data = tfds.as_numpy(tfds.load('imdb_reviews', split="train"))
for item in train_data:
    sentence = str(item['text'].decode('UTF-8').lower())
    soup = BeautifulSoup(sentence)
    sentence = soup.get_text()
    words = sentence.split()
    filtered_sentence = ""
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filtered_sentence = filtered_sentence + word + " "
    imdb_sentences.append(filtered_sentence)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=25000)
tokenizer.fit_on_texts(imdb_sentences)
sequences = tokenizer.texts_to_sequences(imdb_sentences)
print(tokenizer.word_index)
```

所有的句子在处理之前都会转换为小写，因为所有的停用词都存储在小写中。当你打印出你的单词索引时，你会看到这样：

```
{'movie': 1, 'film': 2, 'not': 3, 'one': 4, 'like': 5, 'just': 6, 'good': 7, 
 'even': 8, 'no': 9, 'time': 10, 'really': 11, 'story': 12, 'see': 13, 
 'can': 14, 'much': 15, ...}
```

您可以看到，这比以前要干净得多。然而，总有改进的余地，当我查看完整的索引时，我注意到末尾的一些不太常见的单词是荒谬的。通常评论者会组合词语，例如用破折号（“annoying-conclusion”）或斜杠（“him/her”），而去除标点符号会错误地将它们转换为单个单词。您可以通过在创建句子后立即添加一些代码来避免这种情况，在这里我添加了以下内容：

```
sentence = sentence.replace(`"``,``"`, `"` `,` `"`)
sentence = sentence.replace(`"``.``"`, `"` `.` `"`)
sentence = sentence.replace(`"``-``"`, `"` `-` `"`)
sentence = sentence.replace(`"``/``"`, `"` `/` `"`)
```

这将组合词如“him/her”转换为“him / her”，然后将“/”去除并分词为两个单词。这可能会在后续的训练结果中带来更好的效果。

现在您有了语料库的分词器，您可以对句子进行编码。例如，我们之前在本章中看到的简单句子将会像这样输出：

```
sentences = [
    `'``Today is a sunny day``'`,
    `'``Today is a rainy day``'`,
    `'``Is it sunny today?``'`
]
sequences = tokenizer.texts_to_sequences(sentences)
`print`(sequences)

[[`516`, `5229`, `147`], [`516`, `6489`, `147`], [`5229`, `516`]]
```

如果你解码它们，你会发现停用词被删除了，你得到的句子编码为“今天晴天”，“今天雨天”，和“晴天今天”。

如果你想在代码中实现这一点，你可以创建一个新的`dict`，将键和值颠倒（即，对于单词索引中的键/值对，将值作为键，将键作为值），然后从中进行查找。以下是代码：

```
reverse_word_index = dict(
    [(value, key) for (key, value) in tokenizer.word_index.items()])

decoded_review = ' '.join([reverse_word_index.get(i, '?') for i in sequences[0]])

print(decoded_review)
```

这将产生以下结果：

```
today sunny day
```

### 使用 IMDb 子词数据集

TFDS 还包含几个使用子词预处理的 IMDb 数据集。在这里，您无需按单词分割句子；它们已经被分割成子词。使用子词是在将语料库分割成单个字母（具有较低语义意义的相对较少标记）和单个单词（具有高语义意义的许多标记）之间的一种折衷方法，这种方法通常非常有效地用于语言分类器的训练。这些数据集还包括用于分割和编码语料库的编码器和解码器。

要访问它们，可以调用`tfds.load`并像这样传递`imdb_reviews/subwords8k`或`imdb_reviews/subwords32k`：

```
(train_data, test_data), info = tfds.load(
    `'``imdb_reviews/subwords8k``'`, 
    split = (tfds.`Split`.TRAIN, tfds.`Split`.TEST),
    as_supervised=`True`,
    with_info=`True`
)
```

您可以像这样访问`info`对象上的编码器。这将帮助您查看`vocab_size`：

```
encoder = info.features[`'``text``'`].encoder
`print` (`'``Vocabulary size: {}``'`.format(encoder.vocab_size))
```

这将输出`8185`，因为在此实例中，词汇表由 8,185 个标记组成。如果要查看子词列表，可以使用`encoder.subwords`属性获取它：

```
`print`(encoder.subwords)

[`'``the_``'`, `'``,` `'`, `'``.` `'`, `'``a_``'`, `'``and_``'`, `'``of_``'`, `'``to_``'`, `'``s_``'`, `'``is_``'`, `'``br``'`, `'``in_``'`, `'``I_``'`, 
 `'``that_``'`,...]
```

在这里你可能注意到的一些事情是停用词、标点和语法都在语料库中，HTML 标签如`<br>`也在其中。空格用下划线表示，所以第一个标记是单词“the”。

如果您想要编码一个字符串，可以像这样使用编码器：

```
`sample_string` `=` 'Today is a sunny day'

`encoded_string` `=` `encoder``.``encode``(``sample_string``)`
`print` `(`'Encoded string is {}'`.``format``(``encoded_string``)``)`
```

这将输出一个标记列表：

```
`Encoded`  `string`  `is` `[`6427`,` 4869`,` 9`,` 4`,` 2365`,` 1361`,` 606`]`
```

因此，您的五个单词被编码为七个标记。要查看标记，可以使用编码器的`subwords`属性，它返回一个数组。它是从零开始的，因此“Tod”在“Today”中被编码为`6427`，它是数组中的第 6,426 项：

```
`print``(``encoder``.``subwords``[`6426`]``)`
`Tod`
```

如果您想要解码，可以使用编码器的`decode`方法：

```
encoded_string = encoder.encode(sample_string)

original_string = encoder.decode(encoded_string)
test_string = encoder.decode([`6427`, `4869`, `9`, `4`, `2365`, `1361`, `606`])
```

由于其名称，后几行将具有相同的结果，尽管`encoded_string`是一个标记列表，就像在下一行上硬编码的那样。

## 从 CSV 文件获取文本

虽然 TFDS 拥有大量优秀的数据集，但并非涵盖一切，通常您需要自行加载数据。自然语言处理数据最常见的格式之一是 CSV 文件。在接下来的几章中，您将使用我从开源[文本情感分析数据集](https://oreil.ly/QMMwV)中调整的 Twitter 数据的 CSV 文件。您将使用两个不同的数据集，一个将情感减少为“positive”或“negative”以进行二元分类，另一个使用完整的情感标签范围。每个的结构都是相同的，因此我只会在此处显示二元版本。

Python 的`csv`库使处理 CSV 文件变得简单。在这种情况下，数据存储为每行两个值。第一个是数字（0 或 1），表示情感是否为负面或正面。第二个是包含文本的字符串。

下面的代码将读取 CSV 文件，并对我们在前一节中看到的类似预处理进行处理。它在复合词中的标点周围添加空格，使用`BeautifulSoup`去除 HTML 内容，然后移除所有标点符号：

```
`import` csv
sentences=[]
labels=[]
`with` open(`'``/tmp/binary-emotion.csv``'`, encoding=`'``UTF-8``'`) `as` csvfile:
    reader = csv.reader(csvfile, delimiter=`"``,``"`)
    `for` row `in` reader:
        labels.append(`int`(row[`0`]))
        sentence = row[`1`].lower()
        sentence = sentence.replace(`"``,``"`, `"` `,` `"`)
        sentence = sentence.replace(`"``.``"`, `"` `.` `"`)
        sentence = sentence.replace(`"``-``"`, `"` `-` `"`)
        sentence = sentence.replace(`"``/``"`, `"` `/` `"`)
        soup = `BeautifulSoup`(sentence)
        sentence = soup.get_text()
        words = sentence.split()
        filtered_sentence = `"``"`
        `for` word `in` words:
            word = word.translate(table)
            `if` word `not` `in` stopwords:
                filtered_sentence = filtered_sentence + word + `"` `"`
        sentences.append(filtered_sentence)
```

这将为您提供一个包含 35,327 句子的列表。

### 创建训练和测试子集

现在文本语料库已经被读入句子列表中，您需要将其拆分为训练和测试子集以训练模型。例如，如果您想使用 28,000 个句子进行训练，并将其余部分保留用于测试，您可以使用如下代码：

```
training_size = `28000`

training_sentences = sentences[`0`:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[`0`:training_size]
testing_labels = labels[training_size:]
```

现在您有了一个训练集，您需要从中创建单词索引。以下是使用标记器创建最多 20,000 个单词的词汇表的代码。我们将句子的最大长度设置为 10 个单词，通过截断更长的句子来结束，通过在末尾填充较短的句子，并使用“<OOV>”：

```
`vocab_size` `=` 20000
`max_length` `=` 10
`trunc_type``=``'``post``'`
`padding_type``=``'``post``'`
`oov_tok` `=` `"``<OOV>``"`

`tokenizer` `=` `Tokenizer``(``num_words``=``vocab_size``,` `oov_token``=``oov_tok``)`
`tokenizer``.``fit_on_texts``(``training_sentences``)`

`word_index` `=` `tokenizer``.``word_index`

`training_sequences` `=` `tokenizer``.``texts_to_sequences``(``training_sentences``)`

`training_padded` `=` `pad_sequences``(``training_sequences``,` `maxlen``=``max_length``,` 
                               padding=padding_type, 
                                truncating=trunc_type)
```

您可以通过查看`training_sequences`和`training_padded`来检查结果。例如，在这里我们打印训练序列的第一项，您可以看到它是如何被填充到最大长度 10 的：

```
`print`(training_sequences[`0`])
`print`(training_padded[`0`])

[`18`, `3257`, `47`, `4770`, `613`, `508`, `951`, `423`]
[  `18` `3257`   `47` `4770`  `613`  `508`  `951`  `423`    `0`    `0`]
```

您也可以通过打印来检查单词索引：

```
`{`'<OOV>'`:` `1``,` 'just'`:` `2``,` 'not'`:` `3``,` 'now'`:` `4``,` 'day'`:` `5``,` 'get'`:` `6``,` 'no'`:` `7``,` 
  'good'`:` `8``,` 'like'`:` `9``,` 'go'`:` `10``,` 'dont'`:` `11``,` `.``.``.``}`
```

这里有很多词语你可能想要考虑作为停用词去掉，比如“like”和“dont”。检查词索引总是很有用的。

## 从 JSON 文件获取文本

另一种非常常见的文本文件格式是 JavaScript 对象表示法（JSON）。这是一种开放标准的文件格式，通常用于数据交换，特别是与 Web 应用程序的交互。它易于人类阅读，并设计为使用名称/值对。因此，它特别适合用于标记文本。在 Kaggle 数据集的快速搜索中，JSON 的结果超过 2,500 个。像斯坦福问答数据集（SQuAD）这样的流行数据集，例如，存储在 JSON 中。

JSON 有一个非常简单的语法，对象被包含在大括号中，作为以逗号分隔的名称/值对。例如，代表我的名字的 JSON 对象将是：

```
`{`"firstName" `:` "Laurence"`,`
  "lastName" `:` "Moroney"`}`
```

JSON 还支持数组，这些数组非常类似于 Python 列表，并且由方括号语法表示。这里是一个例子：

```
[
 {`"``firstName``"` : `"``Laurence``"`,
 `"``lastName``"` : `"``Moroney``"`},
 {`"``firstName``"` : `"``Sharon``"`,
 `"``lastName``"` : `"``Agathon``"`}
]
```

对象也可以包含数组，因此这是完全有效的 JSON：

```
[
 {`"``firstName``"` : `"``Laurence``"`,
 `"``lastName``"` : `"``Moroney``"`,
 `"``emails``"`: [`"``lmoroney@gmail.com``"`, `"``lmoroney@galactica.net``"`]
 },
 {`"``firstName``"` : `"``Sharon``"`,
 `"``lastName``"` : `"``Agathon``"`,
 `"``emails``"`: [`"``sharon@galactica.net``"`, `"``boomer@cylon.org``"`]
 }
]
```

一个存储在 JSON 中并且非常有趣的小数据集是由[Rishabh Misra](https://oreil.ly/wZ3oD)创建的用于讽刺检测的新闻标题数据集，可以在[Kaggle](https://oreil.ly/_AScB)上获取。这个数据集收集了来自两个来源的新闻标题：*The Onion* 提供有趣或讽刺的标题，*HuffPost* 提供正常的标题。

讽刺数据集中的文件结构非常简单：

```
{`"``is_sarcastic``"`: `1` `or` `0`, 
 `"``headline``"`: `String` containing headline, 
 `"``article_link``"`: `String` `Containing` link}
```

数据集包含大约 26,000 个项目，每行一个。为了在 Python 中使其更易读，我创建了一个将这些项目封装在数组中的版本，这样它可以作为单个列表进行读取，这在本章的源代码中使用。

### 读取 JSON 文件

Python 的`json`库使得读取 JSON 文件变得简单。鉴于 JSON 使用名称/值对，您可以根据字段的名称索引内容。因此，例如，对于讽刺数据集，您可以创建一个文件句柄到 JSON 文件，使用`json`库打开它，通过迭代逐行读取每个字段，通过字段的名称获取数据项。

下面是代码：

```
`import` json
`with` open(`"``/tmp/sarcasm.json``"`, `'``r``'`) `as` f:
    datastore = json.load(f)
    `for` item `in` datastore:
        sentence = item[`'``headline``'`].lower()
        label= item[`'``is_sarcastic``'`]
        link = item[`'``article_link``'`]
```

这使得创建句子和标签列表变得简单，就像您在整个本章中所做的那样，并对句子进行分词。您还可以在阅读句子时动态进行预处理，删除停用词，HTML 标签，标点符号等。以下是创建句子、标签和 URL 列表的完整代码，同时清理了不需要的词语和字符：

```
`with` `open``(`"/tmp/sarcasm.json"`,` 'r'`)` `as` `f``:`
 `datastore` `=` `json``.``load``(``f``)`

`sentences` `=` `[``]` 
`labels` `=` `[``]`
`urls` `=` `[``]`
`for` `item` `in` `datastore``:`
 `sentence` `=` `item``[`'headline'`]``.``lower``(``)`
 `sentence` `=` `sentence``.``replace``(`","`,` " , "`)`
 `sentence` `=` `sentence``.``replace``(`"."`,` " . "`)`
 `sentence` `=` `sentence``.``replace``(`"-"`,` " - "`)`
 `sentence` `=` `sentence``.``replace``(`"/"`,` " / "`)`
 `soup` `=` `BeautifulSoup``(``sentence``)`
 `sentence` `=` `soup``.``get_text``(``)`
 `words` `=` `sentence``.``split``(``)`
 `filtered_sentence` `=` ""
   `for` `word` `in` `words``:`
 `word` `=` `word``.``translate``(``table``)`
  `if` `word` `not`  `in` `stopwords``:`
 `filtered_sentence` `=` `filtered_sentence` `+` `word` `+` " "
 `sentences``.``append``(``filtered_sentence``)`
 `labels``.``append``(``item``[`'is_sarcastic'`]``)`
 `urls``.``append``(``item``[`'article_link'`]``)`
```

与以前一样，这些可以分为训练集和测试集。如果您想要使用数据集中的 26,000 项中的 23,000 项进行训练，可以执行以下操作：

```
`training_size` `=` 23000

`training_sentences` `=` `sentences``[`0`:``training_size``]`
`testing_sentences` `=` `sentences``[``training_size``:``]`
`training_labels` `=` `labels``[`0`:``training_size``]`
`testing_labels` `=` `labels``[``training_size``:``]`
```

为了对数据进行分词并准备好进行训练，您可以采用与之前相同的方法。在这里，我们再次指定词汇量为 20,000 个词，最大序列长度为 10，末尾截断和填充，并使用“<OOV>”作为 OOV 标记：

```
vocab_size = `20000`
max_length = `10`
trunc_type=`'``post``'`
padding_type=`'``post``'`
oov_tok = `"``<OOV>``"`

tokenizer = `Tokenizer`(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(training_sequences, padding=`'``post``'`)
`print`(word_index)
```

输出将按单词频率顺序排列整个索引：

```
`{`'<OOV>'`:` `1``,` 'new'`:` `2``,` 'trump'`:` `3``,` 'man'`:` `4``,` 'not'`:` `5``,` 'just'`:` `6``,` 'will'`:` `7``,`  
  'one'`:` `8``,` 'year'`:` `9``,` 'report'`:` `10``,` 'area'`:` `11``,` 'donald'`:` `12``,` `.``.``.` `}`
```

希望类似的代码能帮助您看到在准备文本供神经网络分类或生成时可以遵循的模式。在下一章中，您将看到如何使用嵌入来构建文本分类器，在第七章中，您将进一步探索，探讨循环神经网络。然后，在第八章中，您将看到如何进一步增强序列数据以创建能够生成新文本的神经网络！

# 总结

在前面的章节中，您使用图像构建了一个分类器。图像本质上是高度结构化的。您知道它们的尺寸。您知道格式。另一方面，文本可能要复杂得多。它经常是非结构化的，可能包含不想要的内容，比如格式化指令，不总是包含您想要的内容，通常必须进行过滤以去除荒谬或无关的内容。在本章中，您学习了如何使用单词分词将文本转换为数字，并探讨了如何阅读和过滤各种格式的文本。有了这些技能，您现在已经准备好迈出下一步，学习如何从单词中推断*含义*——这是理解自然语言的第一步。
