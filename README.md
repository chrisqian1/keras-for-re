# keras-for-re:relation extraction using keras

### Introduction
This repository is for semantic relation classification task using various models based on keras[9]. It reads relation mention information from instance-level file, like semeval-2010 task 8[1]. The project prepares the input for various models, train and test on data using these models supported in keras. Models available now are Lstm, Attention Lstm, Piecewise Cnn, Bert and XLnet. Other models will be added in the future.

### Model setting
For Lstm[2], Att_Lstm[3] and PCnn[4] model, the main feature is pre-trained word vectors (Glove[5] or Word2Vec[6]). For Lstm and Att_Lstm, additional features include POS and character sequence and etc. For PCnn, additional features include entity positions and etc.

For Bert[7] and XLnet[8], BERT/XLnet tokenizer is used to tokenize a sentence into word pieces. Bert/XLnet pre-trained model is loaded as initial model.

When training, the best model on the validation set will be saved as the final model. Category accuracy will be used to choose the best model by evaluating on the validation set.

### Usage and Example
Training and testing are performed on the semeval 2010 task 8 [1]. To train and test, using the following command line:
```shell
python redoc.py --operation tv --dataset data --modelname Lstm --batchsize 32 --epochs 20 --times 5 --experiment 0
```
option: t for train and v for validate(test), required

dataset: folder of the dataset, required

modelname: Lstm, Att_Lstm, Cnn, Bert and Xlnet, required

batchsize: batch size used when training, default 32. It should be 8 or less in Bert and xlnet when considering the capacity of GPU.

epochs: epochs used when training, default 20.

times: total times of training, default 5

experiment: No. of experiment

### Experimental results on the semeval2010 disease corpus
| Model    | Precision | Recall | F1    |
| -------- | --------- | ------ | ----- |
| Lstm     | 78.20     | 79.31  | 78.54 |
| Att_Lstm | 80.89     | 80.87  | 80.74 |
| PCnn     | 76.28     | 78.10  | 76.94 |
| Bert     | 86.90     | 88.01  | 87.36 |
| Xlnet | 85.25    | 87.73  | 86.34 |

### Reference


[1] Hendrickx, S.N. Kim, Z. Kozareva, P. Nakov,D.´O S´eaghdha, S. Pad´o, M. Pennacchiotti, L. Ro-mano, and S. Szpakowicz. 2010. Semeval-2010 task8: Multi-way classiﬁcation of semantic relations be-tween pairs of nominals. In Proceedings of the 5thInternational Workshop on Semantic Evaluation.

[2] Shu Zhang, Dequan Zheng, Xinchen Hu, and Ming Yang. 2015. Bidirectional long short-term memory networks for relation classification.

[3] Peng Zhou, Wei Shi, Jun Tian, Zhenyu Qi, Bingchen Li, Hongwei Hao, and Bo Xu. 2016. Attention-based bidirectional long short-term memory networks for relation classification. In The 54th Annual Meeting of the Association for Computational Linguistics, page 207.

[4] Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou, and Jun Zhao. 2014. Relation classification via convolutional deep neural network. In Proceedings of COLING, pages 2335–2344.

[5] Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. Glove: Global vectors for word representation. In Empirical Methods in Natural Language Processing (EMNLP), pages 1532–1543.

[6] Noah A Smith. Contextual word representations: A contextual introduction. arXiv preprint arXiv:1902.06006, 2019.

[7] Shanchan Wu and Yifan He. 2019. Enriching pretrained language model with entity information for relation classification. CoRR, abs/1905.08284.

[8] shish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, pages 6000–6010.

[9]Python packages used: keras: https://github.com/keras-team/keras; keras-bert: https://github.com/CyberZHG/keras-bert; keras-xlnet: https://github.com/CyberZHG/keras-xlnet; keras-xlnet: https://github.com/CyberZHG/keras-xlnet; keras-contrib: https://github.com/keras-team/keras-contrib
