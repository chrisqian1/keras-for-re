# keras-for-re:relation extraction using keras
### Introduction
This repository is for pairwise relation extraction task using various models based on keras. It reads relation mention information from instance-level file, like semeval-2010 task 8. The project prepares the input for various models, train and test on data using these models supported in keras. Models available now are Lstm, Attention Lstm, Piecewise Cnn, Bert and XLnet. Other models will be added in the future.
### Model setting
For Lstm, Att_Lstm and PCnn model, the nltk toolkit is used to split abstracts into sentences and tokenize these sentences into tokens. The main feature is pre-trained word vectors (Glove or Word2Vec). For Lstm and Att_Lstm, additional features include POS and character sequence and etc. For PCnn, additional features include entity positions and etc.

For Bert and XLnet, the nltk toolkit is used to only split sentences and BERT/XLnet tokenizer is used to tokenize a sentence into word pieces. BioBert pre-trained model based on PubMed or XLnet pre-trained model is loaded as initial model.

When training, the best model on the validation set will be saved as the final model. Category accuracy will be used to choose the best model by evaluating on the validation set.
