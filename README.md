# evaluating_embeddding_align

根据OpenEA等实体对齐的表示学习模型产生embeding 统计[Recall], [PE]结果，生成候选实体对集合。

### Requirements 

python 

- [falconn](https://falconn-lib.org/) :  [API文档](https://falconn-lib.org/), [Glove例子](https://github.com/FALCONN-LIB/FALCONN/blob/master/src/examples/glove/glove.py) 
- sklean

### Run

将文件download到59.64.xx.xx服务器上。

运行```lsh_falconn.py``` 里```static_openea_mode()```

三个功能：

- 根据thresdshold统计结果，生成候选对。(注意，这里threshold是Metric Distance在threshold以下)

- 根据top-k统计结果，生成候选对。

- 直接计算hits@k值。

  

  说明：falconn本身计算的Metric只有```euclidean```(欧氏距离)和```inner```(向量内积)两种。如果想用Cosine相似度计算评价，则先通过```L2 Normalization```, 后采用```euclidean```，则余弦距离就和欧几里得距离，余弦相似度三者之间有如下关系：
  
  ![](https://boheimg.com/i/2021/01/07/quc567.png)







