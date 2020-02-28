from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from argparse import ArgumentParser
import random
from ie_model import *
from nerdoc import *

random.seed(1234)
max_seq_len = 128
max_word_len = 30


class nerDocSet():
    def __init__(self, id):
        self.id = id        
        self.doc_dict = {}
        self.X = None
        self.y = None
        self.X_v = None
        self.y_v = None
        self.match_dict = {}    
        self.type_list = []
        
    def load_docset(self):
        """
        load relation mention file
        """

        # read from file
        text_file = self.id + '.wrd'
        text_input = open(text_file, encoding = 'utf-8')
        lines = text_input.readlines()

        # split docs(instances)
        text = []
        for line in lines:
            line = line.strip('\n')
            if len(line) != 0:
                text.append(line)
            else:
                if len(text) != 0:
                    docid = text[0].split('\t')[1]
                    doc = nerDoc(text)
                    doc.docid = docid
                    self.doc_dict[docid] = doc
                text = []
        return
        
    def preprocess_docset(self, tokenizer):
        '''
        get info of the relation mention, split sentence, assign labels and add pos tags
        '''

        for docid in self.doc_dict:
            doc = self.doc_dict[docid]
            doc.generate_sentence()
            doc.tokenize(tokenizer)
            doc.assign_label()
            doc.add_POS()
            if doc.docid == '1':
                print([token.word for token in doc.tokenlist])
                print([token.POS for token in doc.tokenlist])
        return


    def update_dict(self, dict_list):
        """
        update index dicts
        """

        label_dict, word_dict, POS_dict, char_dict = dict_list
        
        for docid in self.doc_dict:
            doc = self.doc_dict[docid]
            # update label dict
            if doc.label not in label_dict:
                label_dict[doc.label] = len(label_dict)
            # update word dict
            if len(word_dict) != 0:
                for token in doc.tokenlist:
                    if token.word not in word_dict:
                        word_dict[token.word] = len(word_dict)
                    #update char dict
                    if len(char_dict) != 0:
                        for char in token.word:
                            if char not in char_dict:
                                char_dict[char] = len(char_dict)    
            # update POS dict
            if len(POS_dict) != 0:
                for token in doc.tokenlist:
                    if token.POS not in POS_dict:
                        POS_dict[token.POS] = len(POS_dict)
        return
    
    def prepare_model_input(self, modelname, tokenizer, dict_list, validate):
        '''
        prepare for the input of the model
        '''
        
        label_dict, word_dict, POS_dict, char_dict = dict_list
        feature1_array, feature2_array, feature3_array, self.y = [], [], [], []
        # randomly select 10% of data as validation set
        if validate: 
            feature1_array_v, feature2_array_v, feature3_array_v, self.y_v = [], [], [], []
            idlist = [int(docid) for docid in self.doc_dict]
            random_list=random.sample(idlist, len(idlist)//10)

        # get features and label from each sentence
        for docid in self.doc_dict:
            doc = self.doc_dict[docid]
            # get word sequence and label sequence from sent
            if 'Xlnet' in modelname:
                # feature1: word piece index; feature2: segment; feature3: memory
                encoded = list(tokenizer.encode(doc.sent))[:max_seq_len]
                feature1 = [3] + encoded + [5] * (max_seq_len - 1 - len(encoded))
                feature2 = [0]*len(feature1)
                feature3 = [0]
            elif 'Bert' in modelname:
                # feature1: word piece index; feature2: segment; feature3: entity position(for R_Bert)
                index, segment = tokenizer.encode(doc.sent, max_len = max_seq_len)
                feature1 = [item for item in index]
                feature2 = [item for item in segment]
                feature3 = [doc.e1spos - 1, doc.e1epos, doc.e2spos - 1, doc.e2epos]
            elif 'Cnn' in modelname:
                # feature1: word index; feature2: entity position(for piecewise Cnn)
                token_seq, POS_seq, char_seq = doc.generate_input_seq(max_seq_len, max_word_len)
                feature1 = [word_dict[token] for token in token_seq]
                feature2 = [doc.e1epos+1, doc.e2epos+1]
                feature3 = [0]*len(feature1)
            else:
                # BiLstm
                # feature1: word index; feature2: POS index; featurw3: character index
                token_seq, POS_seq, char_seq = doc.generate_input_seq(max_seq_len, max_word_len)
                feature1 = [word_dict[token] for token in token_seq]
                feature2 = [POS_dict[POS] for POS in POS_seq]
                feature3 = [[char_dict[char] for char in word] for word in char_seq]

            # get label index
            label_ix = label_dict[doc.label]
            # transfer label index to one hot encoding
            label = to_categorical(label_ix, len(label_dict))

            # add features and label to array
            if validate and int(docid) in random_list:
                feature1_array_v.append(feature1)
                feature2_array_v.append(feature2)
                feature3_array_v.append(feature3)
                self.y_v.append(label)
            else:
                feature1_array.append(feature1)
                feature2_array.append(feature2)
                feature3_array.append(feature3)
                self.y.append(label)

        # transfer data to numpy arrays
        if validate:
            feature1_array_v, feature2_array_v, feature3_array_v, self.y_v = \
                np.array(feature1_array_v), np.array(feature2_array_v), np.array(feature3_array_v), np.array(self.y_v)
            if  'Cnn' in modelname:
                self.X_v = [feature1_array_v, feature2_array_v]
            else:
                self.X_v = [feature1_array_v, feature2_array_v, feature3_array_v]

        feature1_array, feature2_array, feature3_array, self.y = \
            np.array(feature1_array), np.array(feature2_array), np.array(feature3_array), np.array(self.y)
        if 'Cnn' in modelname:
            self.X = [feature1_array, feature2_array]
        else:
            self.X = [feature1_array, feature2_array, feature3_array]
        return

    def train(self, modelname, dict_list, bert_path, glove_path, model_path, validate, batch_size, epoch):
        '''
        train model with data
        '''

        label_dict, word_dict, POS_dict, char_dict = dict_list
        # create initial model according to modelname
        if 'Bert' in modelname:
            model = create_bert_classification_model(bert_path, 'sent', modelname, len(label_dict))
        elif 'Xlnet' in modelname:
            model = create_xlnet_classification_model(batch_size, 'sent', len(label_dict))
        else:
            embed_matrix, embed_dim = load_pretrained_embedding_from_file(glove_path, word_dict)
            if 'Cnn' in modelname:
                model = create_cnn_classification_model('sent', modelname, len(word_dict), len(label_dict), embed_matrix, embed_dim)
            else:
                model = create_lstm_classification_model('sent', modelname, len(word_dict), len(POS_dict), len(label_dict), embed_matrix, embed_dim)

        # use checkpoint to select best model during training
        monitor = 'val_crf_viterbi_accuracy' if 'Crf' in modelname else 'val_categorical_accuracy'
        checkpointer = ModelCheckpoint(filepath=model_path, monitor = monitor, verbose=1, save_best_only=True)
        # use validation set to evaluate checkpoint model
        validation = (self.X_v, self.y_v) if validate else None
        # fit the model
        model.fit(x = self.X, y = self.y, batch_size = batch_size, epochs = epoch, validation_data = validation, callbacks = [checkpointer])
        print('model_saved')
        return
        
    def validate(self, modelname, dict_list, bert_path, glove_path, model_path, output_path, batch_size):
        '''
        test/validate with saved model and evaluate the result
        '''

        label_dict, word_dict, POS_dict, char_dict = dict_list
        # create initial model according to modelname
        if 'Bert' in modelname:
            model = create_bert_classification_model(bert_path, 'sent', modelname, len(label_dict))
        elif 'Xlnet' in modelname:
            model = create_xlnet_classification_model(batch_size, 'sent', len(label_dict))
        else:
            embed_matrix, embed_dim = load_pretrained_embedding_from_file(glove_path, word_dict)
            if 'Cnn' in modelname:
                model = create_cnn_classification_model('sent', modelname, len(word_dict), len(label_dict), embed_matrix, embed_dim)
            else:
                model = create_lstm_classification_model('sent', modelname, len(word_dict), len(POS_dict), len(label_dict), embed_matrix, embed_dim)
        print('create_test_model')

        # load model weights from saved model
        model.load_weights(model_path)
        print('model_load')
        # predict and evaluate-
        if 'Xlnet' in modelname:
            batch_size = 1 
        pre = model.predict(self.X, batch_size=batch_size)
        output = open(output_path, 'w')
        self.evaluate(pre, label_dict, output)
        return

    def evaluate(self, pre, label_dict, output):
        '''
        evaluate the result
        '''
    
        # read entity types in gold file and predict file
        gold_labels = [self.doc_dict[docid].label for docid in self.doc_dict]
        pred_labels = []
        for sent_result in pre:
            for label in label_dict:
                label_ix = label_dict[label]
                if sent_result[label_ix] == np.max(sent_result):
                    pred_label = label
            pred_labels.append(pred_label)
        
        # create and print confusion matrix
        conf_matrix, label_list = self.generate_confusion_matrix(gold_labels, pred_labels, label_dict, output)
        
        # create type index
        type_list = []
        for label in label_list:
            type = label.split('-')[-1]
            if type not in type_list:
                type_list.append(type)
        
        # print table title
        title = '\nType\tGold\tTP+FN\tTP\tP\tR\tF1'
        print(title)
        output.write(title+'\n')
        
        # print prf for each type 
        flag = 0
        for type in type_list:
            gold, tp_fp, tp, p,r,f = self.output_prf(gold_labels, pred_labels, type, output)
            if type != 'None':
                if flag == 0:
                    flag = 1
                    prf = np.array([gold, tp_fp, tp, p,r,f])
                else:
                    prf = np.vstack((prf,[gold, tp_fp, tp, p,r,f]))
        # print macro average prf
        total = ['Total']
        for i in range(6):
            if i<3:
                total.append(str(prf.sum(axis = 0)[i]))
            else:
                total.append(str(round(prf.mean(axis = 0)[i], 2)))
        print('\t'.join(total))
        output.write('\t'.join(total) + '\n')
        return

    def generate_confusion_matrix(self, gold_labels, pred_labels, label_dict, output):
        '''
        create and print confusion matrix
        '''
        
        # create confusion matrix
        l = len(label_dict)
        conf_matrix = np.zeros([l, l])
        for i in range(len(gold_labels)):
            conf_matrix[label_dict[gold_labels[i]]][label_dict[pred_labels[i]]] += 1
            conf_matrix[label_dict[gold_labels[i]]][label_dict[pred_labels[i]]] += 1
        
        # create label list
        labels = list(label_dict.keys())
        labels.remove('None')
        labels.append('None')

        # print confusion matrix
        print(labels)
        print('\t'+'\t'.join(labels))
        for label1 in labels:
            row = [label1]
            for label2 in labels:
                row.append(str(conf_matrix[label_dict[label1]][label_dict[label2]]))
            print('\t'.join(row))
            output.write('\t'.join(row) + '\n')
        return conf_matrix, labels

    def output_prf(self, gold_labels, pred_labels, type_name, output):
        '''
        calculate and print prf
        '''
        
        # calculate tp, tp+fp and gold
        tp = tp_fp = gold = 0
        for i in range(len(gold_labels)):
            if type_name in gold_labels[i] and gold_labels[i] in pred_labels[i]:
                tp += 1
            if type_name in gold_labels[i]:
                gold += 1
            if type_name in pred_labels[i]:
                tp_fp += 1
                
        # calculate p,r,f
        p = round(tp/tp_fp*100, 2) if tp_fp != 0 else 0
        r = round(tp/gold*100, 2) if gold != 0 else 0
        f = round(2*p*r/(p+r), 2) if p != 0 and r != 0 else 0
        
        result = '{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(type_name, gold, tp_fp, tp, p, r, f)
        print(result)
        output.write(result + '\n')
        return gold, tp_fp, tp, p,r,f
            
    
def dict_initialize(modelname):
    if 'Bert' in modelname or 'Xlnet' in modelname:
        return {'None':0}, {}, {}, {}
    else:
        return {'None':0}, {'[pad]':0}, {'[pad]':0}, {'[pad]':0}

    
if __name__ == "__main__":
    # python nerdocset.py --option tv --modelname Att_Lstm --batch_size 32 --epoch 20 --stimes 0 --times 5 --experiment 1
    parser = ArgumentParser()
    parser.add_argument('--option', type=str, help='t | v | tv', default='tv')
    parser.add_argument('--datapath', type=str, help='data root', default='data')
    parser.add_argument('--modelname', type=str, help='Xlnet| Bert | Lstm | Crf', default='Lstm_Crf')
    parser.add_argument('--batch_size', type=int, help='size of a batch', default=32)
    parser.add_argument('--epoch', type=int, help='number of total epoches', default=20)
    parser.add_argument('--stimes', type=int, help='number of experiment times', default=0)
    parser.add_argument('--times', type=int, help='number of experiment times', default=5)
    parser.add_argument('--experiment', type=int, help='Where to store models and results', default=1 )
    opt = parser.parse_args()

    bert_model_path = '../../AnaPython/bert-model/bert-base-uncased/'
    glove_path = '../../AnaPython/glove/glove.6B.100d.txt'
    # initialize dicts
    dict_list = dict_initialize(opt.modelname)
    # initialize tokenizer
    if 'Bert' in opt.modelname:
        token_dict, tokenizer = load_bert_tokenizer(bert_model_path)
    elif 'Xlnet' in opt.modelname:
        tokenizer = load_xlnet_tokenizer()
    else:
        tokenizer = None

    # process train dataset
    train_doc = nerDocSet(id=opt.datapath+'/training')
    train_doc.load_docset()
    train_doc.preprocess_docset(tokenizer)
    train_doc.update_dict(dict_list)
    # process test dataset
    test_doc = nerDocSet(id=opt.datapath+'/test')
    test_doc.load_docset()
    test_doc.preprocess_docset(tokenizer)
    test_doc.update_dict(dict_list)

    # prepare model input
    if os.path.exists(opt.datapath + '/dev.wrd'):
        # process dev dataset if it exists
        valid_doc = nerDocSet(opt.datapath + '/dev')
        valid_doc.load_docset()
        valid_doc.preprocess_docset(tokenizer)
        valid_doc.update_dict(dict_list)
        # use dev set as validation set
        train_doc.prepare_model_input(opt.modelname, tokenizer, dict_list, False)
        valid_doc.prepare_model_input(opt.modelname, tokenizer, dict_list, False)
        train_doc.X_v, train_doc.y_v = valid_doc.X, valid_doc.y
    else:
        train_doc.prepare_model_input(opt.modelname, tokenizer, dict_list, True)
    test_doc.prepare_model_input(opt.modelname, tokenizer, dict_list, False)


    # train and validate according to oprations
    for i in range(opt.stimes, opt.times):
        if 't' in opt.option:
            train_doc.train(opt.modelname, dict_list, bert_model_path, glove_path, 'model{}_{}.hdf5'.format(opt.experiment, i), True, opt.batch_size, opt.epoch)
        if 'v' in opt.option:
            test_doc.validate(opt.modelname, dict_list, bert_model_path, glove_path, 'model{}_{}.hdf5'.format(opt.experiment, i), 'result{}_{}.txt'.format(opt.experiment, i), opt.batch_size)

