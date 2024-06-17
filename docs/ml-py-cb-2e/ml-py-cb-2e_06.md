# 第六章 处理文本

# 6.0 引言

非结构化的文本数据，如书籍内容或推文，既是最有趣的特征来源之一，也是最复杂的处理之一。在本章中，我们将介绍将文本转换为信息丰富特征的策略，并使用一些出色的特征（称为*嵌入*），这些特征在涉及自然语言处理（NLP）的任务中变得日益普遍。

这并不意味着这里涵盖的配方是全面的。整个学术学科都专注于处理文本等非结构化数据。在本章中，我们将涵盖一些常用的技术；掌握这些将为我们的预处理工具箱增添宝贵的工具。除了许多通用的文本处理配方外，我们还将演示如何导入和利用一些预训练的机器学习模型来生成更丰富的文本特征。

# 6.1 清理文本

## 问题

你有一些非结构化文本数据，想要完成一些基本的清理工作。

## 解决方案

在下面的例子中，我们查看三本书的文本，并通过 Python 的核心字符串操作，特别是`strip`、`replace`和`split`，对其进行清理：

```py
# Create text
text_data = ["   Interrobang. By Aishwarya Henriette     ",
             "Parking And Going. By Karl Gautier",
             "    Today Is The night. By Jarek Prakash   "]

# Strip whitespaces
strip_whitespace = [string.strip() for string in text_data]

# Show text
strip_whitespace
```

```py
['Interrobang. By Aishwarya Henriette',
 'Parking And Going. By Karl Gautier',
 'Today Is The night. By Jarek Prakash']
```

```py
# Remove periods
remove_periods = [string.replace(".", "") for string in strip_whitespace]

# Show text
remove_periods
```

```py
['Interrobang By Aishwarya Henriette',
 'Parking And Going By Karl Gautier',
 'Today Is The night By Jarek Prakash']
```

我们还创建并应用了一个自定义转换函数：

```py
# Create function
def capitalizer(string: str) -> str:
    return string.upper()

# Apply function
[capitalizer(string) for string in remove_periods]
```

```py
['INTERROBANG BY AISHWARYA HENRIETTE',
 'PARKING AND GOING BY KARL GAUTIER',
 'TODAY IS THE NIGHT BY JAREK PRAKASH']
```

最后，我们可以使用正则表达式进行强大的字符串操作：

```py
# Import library
import re

# Create function
def replace_letters_with_X(string: str) -> str:
    return re.sub(r"[a-zA-Z]", "X", string)

# Apply function
[replace_letters_with_X(string) for string in remove_periods]
```

```py
['XXXXXXXXXXX XX XXXXXXXXX XXXXXXXXX',
 'XXXXXXX XXX XXXXX XX XXXX XXXXXXX',
 'XXXXX XX XXX XXXXX XX XXXXX XXXXXXX']
```

## 讨论

一些文本数据在用于构建特征或在输入算法之前需要进行基本的清理。大多数基本的文本清理可以使用 Python 的标准字符串操作完成。在实际应用中，我们很可能会定义一个自定义的清理函数（例如`capitalizer`），结合一些清理任务，并将其应用于文本数据。虽然清理字符串可能会删除一些信息，但它使数据更易于处理。字符串具有许多有用的固有方法用于清理和处理；一些额外的例子可以在这里找到：

```py
# Define a string
s = "machine learning in python cookbook"

# Find the first index of the letter "n"
find_n = s.find("n")

# Whether or not the string starts with "m"
starts_with_m = s.startswith("m")

# Whether or not the string ends with "python"
ends_with_python = s.endswith("python")

# Is the string alphanumeric
is_alnum = s.isalnum()

# Is it composed of only alphabetical characters (not including spaces)
is_alpha = s.isalpha()

# Encode as utf-8
encode_as_utf8 = s.encode("utf-8")

# Decode the same utf-8
decode = encode_as_utf8.decode("utf-8")

print(
  find_n,
  starts_with_m,
  ends_with_python,
  is_alnum,
  is_alpha,
  encode_as_utf8,
  decode,
  sep = "|"
)
```

```py
5|True|False|False|False|b'machine learning in python cookbook'|machine learning
  in python cookbook
```

## 参见

+   [Python 正则表达式入门教程](https://oreil.ly/hSqsa)

# 6.2 解析和清理 HTML

## 问题

你有包含 HTML 元素的文本数据，并希望仅提取文本部分。

## 解决方案

使用 Beautiful Soup 广泛的选项集来解析和从 HTML 中提取：

```py
# Load library
from bs4 import BeautifulSoup

# Create some HTML code
html = "<div class='full_name'>"\
       "<span style='font-weight:bold'>Masego"\
       "</span> Azra</div>"

# Parse html
soup = BeautifulSoup(html, "lxml")

# Find the div with the class "full_name", show text
soup.find("div", { "class" : "full_name" }).text
```

```py
'Masego Azra'
```

## 讨论

尽管名字奇怪，Beautiful Soup 是一个功能强大的 Python 库，专门用于解析 HTML。通常 Beautiful Soup 用于实时网页抓取过程中处理 HTML，但我们同样可以使用它来提取静态 HTML 中嵌入的文本数据。Beautiful Soup 的全部操作远超出本书的范围，但即使是我们在解决方案中使用的方法，也展示了使用`find()`方法可以轻松地解析 HTML 并从特定标签中提取信息。

## 参见

+   [Beautiful Soup](https://oreil.ly/vh8h3)

# 6.3 删除标点符号

## 问题

你有一项文本数据的特征，并希望去除标点符号。

## 解决方案

定义一个使用`translate`和标点字符字典的函数：

```py
# Load libraries
import unicodedata
import sys

# Create text
text_data = ['Hi!!!! I. Love. This. Song....',
             '10000% Agree!!!! #LoveIT',
             'Right?!?!']

# Create a dictionary of punctuation characters
punctuation = dict.fromkeys(
  (i for i in range(sys.maxunicode)
  if unicodedata.category(chr(i)).startswith('P')
  ),
  None
)

# For each string, remove any punctuation characters
[string.translate(punctuation) for string in text_data]
```

```py
['Hi I Love This Song', '10000 Agree LoveIT', 'Right']
```

## 讨论

Python 的 `translate` 方法因其速度而流行。在我们的解决方案中，首先我们创建了一个包含所有标点符号字符（按照 Unicode 标准）作为键和 `None` 作为值的字典 `punctuation`。接下来，我们将字符串中所有在 `punctuation` 中的字符翻译为 `None`，从而有效地删除它们。还有更可读的方法来删除标点，但这种有些“hacky”的解决方案具有比替代方案更快的优势。

需要意识到标点包含信息这一事实是很重要的（例如，“对吧？”与“对吧！”）。在需要手动创建特征时，删除标点可能是必要的恶；然而，如果标点很重要，我们应该确保考虑到这一点。根据我们试图完成的下游任务的不同，标点可能包含我们希望保留的重要信息（例如，使用“？”来分类文本是否包含问题）。

# 6.4 文本分词

## 问题

你有一段文本，希望将其分解成单独的单词。

## 解决方案

Python 的自然语言工具包（NLTK）具有强大的文本操作集，包括词分词：

```py
# Load library
from nltk.tokenize import word_tokenize

# Create text
string = "The science of today is the technology of tomorrow"

# Tokenize words
word_tokenize(string)
```

```py
['The', 'science', 'of', 'today', 'is', 'the', 'technology', 'of', 'tomorrow']
```

我们还可以将其分词成句子：

```py
# Load library
from nltk.tokenize import sent_tokenize

# Create text
string = "The science of today is the technology of tomorrow. Tomorrow is today."

# Tokenize sentences
sent_tokenize(string)
```

```py
['The science of today is the technology of tomorrow.', 'Tomorrow is today.']
```

## 讨论

*分词*，尤其是词分词，在清洗文本数据后是一项常见任务，因为它是将文本转换为我们将用来构建有用特征的数据的第一步。一些预训练的自然语言处理模型（如 Google 的 BERT）使用特定于模型的分词技术；然而，在从单词级别获取特征之前，词级分词仍然是一种相当常见的分词方法。

# 6.5 移除停用词

## 问题

给定标记化的文本数据，你希望移除极其常见的单词（例如，*a*、*is*、*of*、*on*），它们的信息价值很小。

## 解决方案

使用 NLTK 的 `stopwords`：

```py
# Load library
from nltk.corpus import stopwords

# You will have to download the set of stop words the first time
# import nltk
# nltk.download('stopwords')

# Create word tokens
tokenized_words = ['i',
                   'am',
                   'going',
                   'to',
                   'go',
                   'to',
                   'the',
                   'store',
                   'and',
                   'park']

# Load stop words
stop_words = stopwords.words('english')

# Remove stop words
[word for word in tokenized_words if word not in stop_words]
```

```py
['going', 'go', 'store', 'park']
```

## 讨论

虽然“停用词”可以指任何我们希望在处理前移除的单词集，但通常这个术语指的是那些本身包含很少信息价值的极其常见的单词。是否选择移除停用词将取决于你的具体用例。NLTK 有一个常见停用词列表，我们可以用来查找并移除我们标记化的单词中的停用词：

```py
# Show stop words
stop_words[:5]
```

```py
['i', 'me', 'my', 'myself', 'we']
```

注意，NLTK 的 `stopwords` 假定标记化的单词都是小写的。

# 6.6 词干提取

## 问题

你有一些标记化的单词，并希望将它们转换为它们的根形式。

## 解决方案

使用 NLTK 的 `PorterStemmer`：

```py
# Load library
from nltk.stem.porter import PorterStemmer

# Create word tokens
tokenized_words = ['i', 'am', 'humbled', 'by', 'this', 'traditional', 'meeting']

# Create stemmer
porter = PorterStemmer()

# Apply stemmer
[porter.stem(word) for word in tokenized_words]
```

```py
['i', 'am', 'humbl', 'by', 'thi', 'tradit', 'meet']
```

## 讨论

*词干提取* 通过识别和移除词缀（例如动名词），将单词减少到其词干，同时保持单词的根本含义。例如，“tradition” 和 “traditional” 都有 “tradit” 作为它们的词干，表明虽然它们是不同的词，但它们代表同一个一般概念。通过词干提取我们的文本数据，我们将其转换为不太可读但更接近其基本含义的形式，因此更适合跨观察进行比较。NLTK 的 `PorterStemmer` 实现了广泛使用的 Porter 词干提取算法，以移除或替换常见的后缀，生成词干。

## 参见

+   [Porter 词干提取算法](https://oreil.ly/Z4NTp)

# 6.7 标记词性

## 问题

您拥有文本数据，并希望标记每个单词或字符的词性。

## 解决方案

使用 NLTK 的预训练词性标注器：

```py
# Load libraries
from nltk import pos_tag
from nltk import word_tokenize

# Create text
text_data = "Chris loved outdoor running"

# Use pretrained part of speech tagger
text_tagged = pos_tag(word_tokenize(text_data))

# Show parts of speech
text_tagged
```

```py
[('Chris', 'NNP'), ('loved', 'VBD'), ('outdoor', 'RP'), ('running', 'VBG')]
```

输出是一个包含单词和词性标签的元组列表。NLTK 使用宾树库的词性标签。宾树库的一些示例标签包括：

| Tag | 词性 |
| --- | --- |
| NNP | 专有名词，单数 |
| NN | 名词，单数或集合名词 |
| RB | 副词 |
| VBD | 动词，过去式 |
| VBG | 动词，动名词或现在分词 |
| JJ | 形容词 |
| PRP | 人称代词 |

一旦文本被标记，我们可以使用标签找到特定的词性。例如，这里是所有的名词：

```py
# Filter words
[word for word, tag in text_tagged if tag in ['NN','NNS','NNP','NNPS'] ]
```

```py
['Chris']
```

更现实的情况可能是有数据，每个观察都包含一条推文，并且我们希望将这些句子转换为各个词性的特征（例如，如果存在专有名词，则为 `1`，否则为 `0`）：

```py
# Import libraries
from sklearn.preprocessing import MultiLabelBinarizer

# Create text
tweets = ["I am eating a burrito for breakfast",
          "Political science is an amazing field",
          "San Francisco is an awesome city"]

# Create list
tagged_tweets = []

# Tag each word and each tweet
for tweet in tweets:
    tweet_tag = nltk.pos_tag(word_tokenize(tweet))
    tagged_tweets.append([tag for word, tag in tweet_tag])

# Use one-hot encoding to convert the tags into features
one_hot_multi = MultiLabelBinarizer()
one_hot_multi.fit_transform(tagged_tweets)
```

```py
array([[1, 1, 0, 1, 0, 1, 1, 1, 0],
       [1, 0, 1, 1, 0, 0, 0, 0, 1],
       [1, 0, 1, 1, 1, 0, 0, 0, 1]])
```

使用 `classes_`，我们可以看到每个特征都是一个词性标签：

```py
# Show feature names
one_hot_multi.classes_
```

```py
array(['DT', 'IN', 'JJ', 'NN', 'NNP', 'PRP', 'VBG', 'VBP', 'VBZ'], dtype=object)
```

## 讨论

如果我们的文本是英语且不涉及专业主题（例如医学），最简单的解决方案是使用 NLTK 的预训练词性标注器。但是，如果 `pos_tag` 不太准确，NLTK 还为我们提供了训练自己标注器的能力。训练标注器的主要缺点是我们需要一个大型文本语料库，其中每个词的标签是已知的。构建这种标记语料库显然是劳动密集型的，可能是最后的选择。

## 参见

+   [宾树库项目中使用的词性标签的字母顺序列表](https://oreil.ly/31xKf)

# 6.8 执行命名实体识别

## 问题

您希望在自由文本中执行命名实体识别（如“人物”，“州”等）。

## 解决方案

使用 spaCy 的默认命名实体识别管道和模型从文本中提取实体：

```py
# Import libraries
import spacy

# Load the spaCy package and use it to parse the text
# make sure you have run "python -m spacy download en"
nlp = spacy.load("en_core_web_sm")
doc = nlp("Elon Musk offered to buy Twitter using $21B of his own money.")

# Print each entity
print(doc.ents)

# For each entity print the text and the entity label
for entity in doc.ents:
    print(entity.text, entity.label_, sep=",")
```

```py
(Elon Musk, Twitter, 21B)
Elon Musk, PERSON
Twitter, ORG
21B, MONEY
```

## 讨论

命名实体识别是从文本中识别特定实体的过程。像 spaCy 这样的工具提供预配置的管道，甚至是预训练或微调的机器学习模型，可以轻松识别这些实体。在本例中，我们使用 spaCy 识别文本中的人物（“Elon Musk”）、组织（“Twitter”）和金额（“21B”）。利用这些信息，我们可以从非结构化文本数据中提取结构化信息。这些信息随后可以用于下游机器学习模型或数据分析。

训练自定义命名实体识别模型超出了本示例的范围；但是，通常使用深度学习和其他自然语言处理技术来完成此任务。

## 另请参阅

+   [spaCy 命名实体识别文档](https://oreil.ly/cN8KM)

+   [命名实体识别，维基百科](https://oreil.ly/G8WDF)

# 6.9 将文本编码为词袋模型

## 问题

您有文本数据，并希望创建一组特征，指示观察文本中包含特定单词的次数。

## 解决方案

使用 scikit-learn 的`CountVectorizer`：

```py
# Load library
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

# Create text
text_data = np.array(['I love Brazil. Brazil!',
                      'Sweden is best',
                      'Germany beats both'])

# Create the bag of words feature matrix
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)

# Show feature matrix
bag_of_words
```

```py
<3x8 sparse matrix of type '<class 'numpy.int64'>'
    with 8 stored elements in Compressed Sparse Row format>
```

此输出是一个稀疏数组，在我们有大量文本时通常是必要的。但是，在我们的玩具示例中，我们可以使用`toarray`查看每个观察结果的单词计数矩阵：

```py
bag_of_words.toarray()
```

```py
array([[0, 0, 0, 2, 0, 0, 1, 0],
       [0, 1, 0, 0, 0, 1, 0, 1],
       [1, 0, 1, 0, 1, 0, 0, 0]], dtype=int64)
```

我们可以使用`get_feature_names`方法查看与每个特征关联的单词：

```py
# Show feature names
count.get_feature_names_out()
```

```py
array(['beats', 'best', 'both', 'brazil', 'germany', 'is', 'love',
       'sweden'], dtype=object)
```

注意，`I`从`I love Brazil`中不被视为一个标记，因为默认的`token_pattern`只考虑包含两个或更多字母数字字符的标记。

然而，这可能会令人困惑，为了明确起见，这里是特征矩阵的外观，其中单词作为列名（每行是一个观察结果）：

| beats | best | both | brazil | germany | is | love | sweden |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 2 | 0 | 0 | 1 | 0 |
| 0 | 1 | 0 | 0 | 0 | 1 | 0 | 1 |
| 1 | 0 | 1 | 0 | 1 | 0 | 0 | 0 |

## 讨论

将文本转换为特征的最常见方法之一是使用词袋模型。词袋模型为文本数据中的每个唯一单词输出一个特征，每个特征包含在观察中出现的次数计数。例如，在我们的解决方案中，句子“I love Brazil. Brazil!”中，“brazil”特征的值为`2`，因为单词*brazil*出现了两次。

我们解决方案中的文本数据故意很小。在现实世界中，文本数据的单个观察结果可能是整本书的内容！由于我们的词袋模型为数据中的每个唯一单词创建一个特征，因此生成的矩阵可能包含数千个特征。这意味着矩阵的大小有时可能会在内存中变得非常大。幸运的是，我们可以利用词袋特征矩阵的常见特性来减少我们需要存储的数据量。

大多数单词可能不会出现在大多数观察中，因此单词袋特征矩阵将主要包含值为 0 的值。我们称这些类型的矩阵为 *稀疏*。我们可以只存储非零值，然后假定所有其他值为 0，以节省内存，特别是在具有大型特征矩阵时。`CountVectorizer` 的一个好处是默认输出是稀疏矩阵。

`CountVectorizer` 配备了许多有用的参数，使得创建单词袋特征矩阵变得容易。首先，默认情况下，每个特征是一个单词，但这并不一定是情况。我们可以将每个特征设置为两个单词的组合（称为 2-gram）甚至三个单词（3-gram）。`ngram_range` 设置了我们的*n*-gram 的最小和最大大小。例如，`(2,3)` 将返回所有的 2-gram 和 3-gram。其次，我们可以使用 `stop_words` 轻松地去除低信息的填充词，可以使用内置列表或自定义列表。最后，我们可以使用 `vocabulary` 限制我们希望考虑的单词或短语列表。例如，我们可以仅为国家名称的出现创建一个单词袋特征矩阵：

```py
# Create feature matrix with arguments
count_2gram = CountVectorizer(ngram_range=(1,2),
                              stop_words="english",
                              vocabulary=['brazil'])
bag = count_2gram.fit_transform(text_data)

# View feature matrix
bag.toarray()
```

```py
array([[2],
       [0],
       [0]])
```

```py
# View the 1-grams and 2-grams
count_2gram.vocabulary_
```

```py
{'brazil': 0}
```

## 另请参阅

+   [*n*-gram, 维基百科](https://oreil.ly/XWIrM)

+   [袋装爆米花遇到袋装爆米花](https://oreil.ly/IiyRV)

# 6.10 加权词重要性

## 问题

您希望一个单词袋，其中单词按其对观察的重要性加权。

## 解决方案

通过使用词频-逆文档频率（<math display="inline"><mtext class="left_paren" fontstyle="italic">tf-idf</mtext></math>）比较单词在文档（推文、电影评论、演讲文稿等）中的频率与单词在所有其他文档中的频率。scikit-learn 通过 `TfidfVectorizer` 轻松实现这一点：

```py
# Load libraries
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Create text
text_data = np.array(['I love Brazil. Brazil!',
                      'Sweden is best',
                      'Germany beats both'])

# Create the tf-idf feature matrix
tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(text_data)

# Show tf-idf feature matrix
feature_matrix
```

```py
<3x8 sparse matrix of type '<class 'numpy.float64'>'
    with 8 stored elements in Compressed Sparse Row format>
```

就像在食谱 6.9 中一样，输出是一个稀疏矩阵。然而，如果我们想将输出视为密集矩阵，我们可以使用 `toarray`：

```py
# Show tf-idf feature matrix as dense matrix
feature_matrix.toarray()
```

```py
array([[ 0\.        ,  0\.        ,  0\.        ,  0.89442719,  0\.        ,
         0\.        ,  0.4472136 ,  0\.        ],
       [ 0\.        ,  0.57735027,  0\.        ,  0\.        ,  0\.        ,
         0.57735027,  0\.        ,  0.57735027],
       [ 0.57735027,  0\.        ,  0.57735027,  0\.        ,  0.57735027,
         0\.        ,  0\.        ,  0\.        ]])
```

`vocabulary_` 展示了每个特征的词汇：

```py
# Show feature names
tfidf.vocabulary_
```

```py
{'love': 6,
 'brazil': 3,
 'sweden': 7,
 'is': 5,
 'best': 1,
 'germany': 4,
 'beats': 0,
 'both': 2}
```

## 讨论

单词在文档中出现的次数越多，该单词对该文档的重要性就越高。例如，如果单词 *economy* 经常出现，这表明文档可能与经济有关。我们称之为 *词频* (<math display="inline"><mtext class="left_paren" fontstyle="italic">tf</mtext></math>)。

相反，如果一个词在许多文档中出现，它可能对任何单个文档的重要性较低。例如，如果某个文本数据中的每个文档都包含单词 *after*，那么它可能是一个不重要的词。我们称之为 *文档频率* (<math display="inline"><mtext class="left_paren" fontstyle="italic">df</mtext></math>)。

通过结合这两个统计量，我们可以为每个单词分配一个分数，代表该单词在文档中的重要性。具体来说，我们将 <math display="inline"><mtext fontstyle="italic">tf</mtext></math> 乘以文档频率的倒数 <math display="inline"><mtext class="left_paren" fontstyle="italic">idf</mtext></math>：

<math display="block"><mrow><mtext class="left_paren" fontstyle="italic">tf-idf</mtext> <mo>(</mo> <mi>t</mi> <mo>,</mo> <mi>d</mi> <mo>)</mo> <mo>=</mo> <mi>t</mi> <mi>f</mi> <mo class="left_paren">(</mo> <mi>t</mi> <mo>,</mo> <mi>d</mi> <mo>)</mo> <mo>×</mo> <mtext class="left_paren" fontstyle="italic">idf</mtext> <mo>(</mo> <mi class="left_paren">t</mi> <mo>)</mo></mrow></math>

其中 <math display="inline"><mi>t</mi></math> 是一个单词（术语），<math display="inline"><mi>d</mi></math> 是一个文档。关于如何计算 <math display="inline"><mtext fontstyle="italic">tf</mtext></math> 和 <math display="inline"><mtext fontstyle="italic">idf</mtext></math> 有许多不同的变体。在 scikit-learn 中，<math display="inline"><mtext fontstyle="italic">tf</mtext></math> 简单地是单词在文档中出现的次数，<math display="inline"><mtext fontstyle="italic">idf</mtext></math> 计算如下：

<math display="block"><mrow><mtext class="left_paren" fontstyle="italic">idf</mtext> <mrow><mo>(</mo> <mi>t</mi> <mo>)</mo></mrow> <mo>=</mo> <mi>l</mi> <mi>o</mi> <mi>g</mi> <mfrac><mrow><mn>1</mn><mo>+</mo><msub><mi>n</mi> <mi>d</mi></msub></mrow> <mrow><mn>1</mn><mo>+</mo><mtext class="left_paren" fontstyle="italic">df</mtext><mo>(</mo><mi>d</mi><mo>,</mo><mi>t</mi><mo>)</mo></mrow></mfrac> <mo>+</mo> <mn>1</mn></mrow></math>

其中 <math display="inline"><msub><mi>n</mi><mi>d</mi></msub></math> 是文档数量，<math display="inline"><mtext class="left_paren" fontstyle="italic">df</mtext><mo>(</mo><mi>d</mi><mo>,</mo><mi>t</mi><mo>)</mo></math> 是术语 <math display="inline"><mi>t</mi></math> 的文档频率（即术语出现的文档数量）。

默认情况下，scikit-learn 使用欧几里得范数（L2 范数）对 <math display="inline"><mtext fontstyle="italic">tf-idf</mtext></math> 向量进行归一化。结果值越高，单词对文档的重要性越大。

## 另请参阅

+   [scikit-learn 文档: *tf–idf* 术语加权](https://oreil.ly/40WeT)

# 6.11 使用文本向量计算搜索查询中的文本相似度

## 问题

您想要使用 <math display="inline"><mtext fontstyle="italic">tf-idf</mtext></math> 向量来实现 Python 中的文本搜索功能。

## 解决方案

使用 scikit-learn 计算 <math display="inline"><mtext fontstyle="italic">tf-idf</mtext></math> 向量之间的余弦相似度：

```py
# Load libraries
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Create searchable text data
text_data = np.array(['I love Brazil. Brazil!',
                      'Sweden is best',
                      'Germany beats both'])

# Create the tf-idf feature matrix
tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(text_data)

# Create a search query and transform it into a tf-idf vector
text = "Brazil is the best"
vector = tfidf.transform([text])

# Calculate the cosine similarities between the input vector and all other
  vectors
cosine_similarities = linear_kernel(vector, feature_matrix).flatten()

# Get the index of the most relevent items in order
related_doc_indicies = cosine_similarities.argsort()[:-10:-1]

# Print the most similar texts to the search query along with the cosine
  similarity
print([(text_data[i], cosine_similarities[i]) for i in related_doc_indicies])
```

```py
[
  (
    'Sweden is best', 0.6666666666666666),
    ('I love Brazil. Brazil!', 0.5163977794943222),
    ('Germany beats both', 0.0
    )
]
```

## 讨论

文本向量对于诸如搜索引擎之类的 NLP 用例非常有用。计算了一组句子或文档的 <math display="inline"><mtext fontstyle="italic">tf-idf</mtext></math> 向量后，我们可以使用相同的 `tfidf` 对象来向量化未来的文本集。然后，我们可以计算输入向量与其他向量矩阵之间的余弦相似度，并按最相关的文档进行排序。

余弦相似度的取值范围为[0, 1.0]，其中 0 表示最不相似，1 表示最相似。由于我们使用<math display="inline"><mtext fontstyle="italic">tf-idf</mtext></math>向量来计算向量之间的相似度，单词出现的频率也被考虑在内。然而，在一个小的语料库（文档集合）中，即使是“频繁”出现的词语也可能不频繁出现。在这个例子中，“瑞典是最好的”是最相关的文本，与我们的搜索查询“巴西是最好的”最相似。由于查询提到了巴西，我们可能期望“我爱巴西。巴西！”是最相关的；然而，由于“是”和“最好”，“瑞典是最好的”是最相似的。随着我们向语料库中添加的文档数量的增加，不重要的词语将被加权较少，对余弦相似度计算的影响也将减小。

## 参见

+   [余弦相似度，Geeks for Geeks](https://oreil.ly/-5Odv)

+   [Nvidia 给了我一台价值 15000 美元的数据科学工作站——这是我在其中做的事情（用 Python 构建 Pubmed 搜索引擎）](https://oreil.ly/pAxbR)

# 6.12 使用情感分析分类器

## 问题

您希望对一些文本的情感进行分类，以便作为特征或在下游数据分析中使用。

## 解决方案

使用`transformers`库的情感分类器。

```py
# Import libraries
from transformers import pipeline

# Create an NLP pipeline that runs sentiment analysis
classifier = pipeline("sentiment-analysis")

# Classify some text
# (this may download some data and models the first time you run it)
sentiment_1 = classifier("I hate machine learning! It's the absolute worst.")
sentiment_2 = classifier(
    "Machine learning is the absolute"
    "bees knees I love it so much!"
)

# Print sentiment output
print(sentiment_1, sentiment_2)
```

```py
[
  {
    'label': 'NEGATIVE',
    'score': 0.9998020529747009
  }
]
[
  {
    'label': 'POSITIVE',
    'score': 0.9990628957748413
  }
]
```

## 讨论

`transformers`库是一个极为流行的自然语言处理任务库，包含许多易于使用的 API，用于训练模型或使用预训练模型。我们将在第二十二章更详细地讨论 NLP 和这个库，但这个例子作为使用预训练分类器在您的机器学习流水线中生成特征、分类文本或分析非结构化数据的强大工具的高级介绍。

## 参见

+   [Hugging Face Transformers 快速导览](https://oreil.ly/7hT6W)
