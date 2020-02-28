# keras-for-re:relation extraction using keras

### Introduction
This repository is for pairwise relation extraction task using various models based on keras. It reads relation mention information from instance-level file, like semeval-2010 task 8. The project prepares the input for various models, train and test on data using these models supported in keras. Models available now are Lstm, Attention Lstm, Piecewise Cnn, Bert and XLnet. Other models will be added in the future.

### Model setting
For Lstm, Att_Lstm and PCnn model, the nltk toolkit is used to split abstracts into sentences and tokenize these sentences into tokens. The main feature is pre-trained word vectors (Glove or Word2Vec). For Lstm and Att_Lstm, additional features include POS and character sequence and etc. For PCnn, additional features include entity positions and etc.

For Bert and XLnet, the nltk toolkit is used to only split sentences and BERT/XLnet tokenizer is used to tokenize a sentence into word pieces. BioBert pre-trained model based on PubMed or XLnet pre-trained model is loaded as initial model.

When training, the best model on the validation set will be saved as the final model. Category accuracy will be used to choose the best model by evaluating on the validation set.

### Usage and Example
Training and testing are performed on the semeval 2010 task 8 [1]. To train and test, using the following command line:
```shell
python bio_ner.py --operation tv --dataset data --modelname Lstm --batchsize 32 --epochs 20 --times 5 --experiment 0
```
option: t for train and v for validate(test), required

dataset: folder of the dataset, required

modelname: Lstm, Att_Lstm, Cnn, Bert and Xlnet, required

batchsize: batch size used when training, default 32. It should be 8 or less in Bert and xlnet when considering the capacity of GPU.

epochs: epochs used when training, default 20.

times: total times of training, default 5

experiment: No. of experiment

### Experimental results on the NCBI disease corpus
| Model    | Precision | Recall | F1    |
| -------- | --------- | ------ | ----- |
| Lstm     | 78.20     | 79.31  | 78.54 |
| Att_Lstm | 80.89     | 80.87  | 80.74 |
| Bert     | 86.90     | 88.01  | 87.36 |
| Xlnet | 85.25    | 87.73  | 86.34 |

### Reference
[1] Hendrickx, S.N. Kim, Z. Kozareva, P. Nakov,D.´O S´eaghdha, S. Pad´o, M. Pennacchiotti, L. Ro-mano, and S. Szpakowicz. 2010. Semeval-2010 task8: Multi-way classiﬁcation of semantic relations be-tween pairs of nominals. In Proceedings of the 5thInternational Workshop on Semantic Evaluation.
[2] Bryan Rink and Sanda Harabagiu. 2010. Utd: Classifying semantic relations by combining lexical and semantic resources. In Proceedings of the 5th International Workshop on Semantic Evaluation, pages 256–259. Association for Computational Linguistics.
[3] Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou, and Jun Zhao. 2014. Relation classification via convolutional deep neural network. In Proceedings of COLING, pages 2335–2344.
[4] Dongxu Zhang and Dong Wang. 2015. Relation classification via recurrent neural network. arXiv preprint arXiv:1508.01006.
[5] Peng Zhou, Wei Shi, Jun Tian, Zhenyu Qi, Bingchen Li, Hongwei Hao, and Bo Xu. 2016. Attention-based bidirectional long short-term memory networks for relation classification. In The 54th Annual Meeting of the Association for Computational Linguistics, page 207.
[6] Shanchan Wu and Yifan He. 2019. Enriching pretrained language model with entity information for relation classification. CoRR, abs/1905.08284.
[7] Python packages used: keras: https://github.com/keras-team/keras; keras-bert: https://github.com/CyberZHG/keras-bert; keras-xlnet: https://github.com/CyberZHG/keras-xlnet; keras-xlnet: https://github.com/CyberZHG/keras-xlnet; keras-contrib: https://github.com/keras-team/keras-contrib
