from nltk.tokenize import word_tokenize
from nltk import pos_tag


class nerDoc():
    def __init__(self, text):
        self.docid = None
        self.text = text
        self.sent = ''
        self.tokenlist = []
        self.POSlist = []
        self.label = ''
        self.e1spos, self.e1epos = int(text[2].split('\t')[-2]), int(text[2].split('\t')[-1])
        self.e2spos, self.e2epos = int(text[3].split('\t')[-2]), int(text[3].split('\t')[-1])
        
    def generate_sentence(self):
        """
        generate sentence and process mention tag
        """

        title = self.text[0].split('\t')
        sent = []
        for i in range(int(title[-2]), len(self.text)):
            word = self.text[i].split('\t')[-1]
            if word in ['<e1>', '</e1>']:
                word = '#'
            if word in ['<e2>', '</e2>']:
                word = '@'
            sent.append(word)
        self.sent = ' '.join(sent)
        return

    def tokenize(self, tokenizer = None):
        """
        tokenize according to tokenizer
        """

        # use word tokenizer
        if tokenizer != None:
            wordlist = tokenizer.tokenize(self.sent)
        else:
            wordlist = word_tokenize(self.sent)
        # add token to tokenlist
        for word in wordlist:
            token = nerToken(word)
            self.tokenlist.append(token)
        return

    def add_POS(self):
        """
        addpos tags to tokens
        """

        # generate POS list
        word_list = [token.word for token in self.tokenlist if token.word != '#' and token.word != '@']
        POSlist = [POS[1] for POS in pos_tag(word_list)]
        # adjust accroding to entity mention tags
        for i, token in enumerate(self.tokenlist):
            if i == self.e1spos - 2 or i == self.e1epos:
                token.POS = '#'
            elif i == self.e2spos - 2 or i == self.e2epos:
                token.POS = '@'
            elif i < self.e1spos - 2:
                token.POS = POSlist[i]
            elif i < self.e1epos:
                token.POS = POSlist[i-1]
            elif i < self.e2spos - 2:
                token.POS = POSlist[i-2]
            elif i < self.e2epos:
                token.POS = POSlist[i-3]
            else:
                token.POS = POSlist[i-4]
        return

    def assign_label(self):
        """
        assign labels according to the entity position
        """

        typeline = self.text[1].split('\t')
        if len(typeline) == 2:
            self.label = typeline[-1]
        # connect reversed relation with -
        else:
            self.label = '{}-{}'.format(typeline[-2], typeline[-1])
        return

    def generate_input_seq(self, max_seq_len, max_word_len):
        """
        generate feature sequence
        """

        # generate token sequence
        token_seq = [token.word for token in self.tokenlist][:max_seq_len]
        token_seq = token_seq + ['[pad]'] * (max_seq_len - len(token_seq))
        # generate POS sequence
        POS_seq = [token.POS for token in self.tokenlist][:max_seq_len]
        POS_seq = POS_seq + ['[pad]'] * (max_seq_len - len(POS_seq))
        # generate character sequence
        char_seq = []
        for token in self.tokenlist:
            char_list = [char for char in token.word][:max_word_len-4]
            char_list = ['[pad]'] * 2 + char_list + ['[pad]'] * (max_word_len - 2 - len(char_list))
            char_seq .append(char_list)
        char_seq = char_seq[:max_seq_len]
        char_seq = char_seq + [['[pad]' for i in range(max_word_len)]] * (max_seq_len - len(char_seq))

        return token_seq, POS_seq, char_seq
        

class nerToken(object):
    def __init__(self, word):
        self.word = word
        self.POS = ''
