# 任务四：垃圾邮件分类器

<img src="https://github.com/KURIYAMA595/KUKUKU/blob/main/images/Rhine%20Lab.png" alt="没啥用用的图片">


代码部署保存在了：main_classify.ipynb
---

 

## 代码核心功能说明


### 代码主要分为以下几个部分：

1、数据预处理和特征提取（get_words和get_top_words函数）

2、特征向量构建（vector变量）

3、模型训练（MultinomialNB）

4、预测新邮件（predict函数）

### 关键函数解析：

### 在数据预处理和特征提取中

```def get_words(filename):
    words = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
            line = cut(line)
            line = filter(lambda word: len(word) > 1, line)
            words.extend(line)
    return words
```
这个函数主要是用来读取文本文件并进行预处理，其处理步骤是先去除每行首尾空白，然后使用正则表达式移除标点符号和数字，再使用结巴分词进行中文分词，最后过滤掉单字词（长度<=1的词）

```def get_top_words(top_num):
    filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
    for filename in filename_list:
        all_words.append(get_words(filename))
    freq = Counter(chain(*all_words))
    return [i[0] for i in freq.most_common(top_num)]
```
这个函数主要是从所有训练邮件中提取出现频率最高的top_num个词，其处理步骤主要是先读取151封邮件（0.txt到150.txt），然后对每封邮件调用get_words()获取词列表，再使用Counter统计所有词的出现频率，最后返回出现频率最高的top_num个词。


## 特征工程提取：

### 构建词向量

```top_words = get_top_words(100)
vector = []
for words in all_words:
    word_map = list(map(lambda word: words.count(word), top_words))
    vector.append(word_map)
vector = np.array(vector)
```
这个代码首先为每封邮件构建一个100维的特征向量，每个维度对应一个高频词，值为该词在当前邮件中出现的次数，最终得到151×100的特征矩阵（151封邮件，每封邮件100个特征）

### 构建标签
```labels = np.array([1]*127 + [0]*24)
```
前127封邮件标记为垃圾邮件（1）；后24封邮件标记为普通邮件（0）


## 预测新邮件

```def predict(filename):
    """对未知邮件分类"""
    # 构建未知邮件的词向量
    words = get_words(filename)
    current_vector = np.array(
        tuple(map(lambda word: words.count(word), top_words)))
    # 预测结果
    result = model.predict(current_vector.reshape(1, -1))
    return '垃圾邮件' if result == 1 else '普通邮件'
```

这个代码首先对新邮件进行同样的预处理和分词，然后统计top_words中每个词在新邮件中的出现次数，再构建特征向量，最后使用训练好的模型预测类别


### 高频词/TF-IDF特征模式切换方法

### 修改特征提取逻辑

想要实现高频词/TF-IDF特征模式的切换，我们可以在原有代码中增加TF-IDF支持，通过参数控制模式切换

