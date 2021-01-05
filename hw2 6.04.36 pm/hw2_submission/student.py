#!/usr/bin/env python3
"""
#Group Name: zy_zcb
#Gruop Member: Yu Zhang(z5238743), Chengbin Zhang(z5252388)
#Group ID: g023634
#Weighted Score on Vlab (CPU mode): 84.50%

DESCRIPTION:
Abstract:
In this assignment, we have tried several different models, RNN, LSTM, GRU and BiLSTM-Attention. We also want to
use BERT but failed when building models. The final algorithm we use is BiLSTM-Attention as it costs least calculation
resources but performs best among all these algorithms (Although there is not much difference, at most 2 percent...).
Finally we got an accuracy of 84.70% on test case, the result is not satisfactory so we perform data analysis on test case prediction.
The analysis indicts that business category 1 cannot be well distinguished from others, the accuracy for category 1 is below
80% (around 79.5%). Based on more detailed analysis, we find that some key word of category 1 is same with others, for example,
pharmacy can occur in both category 1 and 3, whatsmore, these words contributes most on final prediction. That is to say, 
it is hard for us to improve performance on comments like this.

Preprocessing:
First, We got a list of stop words from: https://gist.github.com/sebleier/554280. And we add some new stopwords to
pertect it. Secondly, After loading the raw sentence(reviewText), we use split(" ") function to divide sentence into
words by split every space. Third, we remove stopwords, punctuations and non-English characters by nominalization.
This is because stopsword are not associated with sentiment expression and if punctuations and non-English characters
exist, it will add more noises and decrease the accuracy.

Building the models:
At the beginning, considering 2 types of classification we made(rating and business_category), we build a LSTM(Long
Short Term Memory) and GRU(Gated Recurrent Unit) for Business category and rating calssification. Because LSTM
is well-suited to learn from experience to classify, process and predict time series when there are very long time lags
of unknownsize between important events. Besides, LSTM process the entire sequence of data. in the project, the rating
classification will combine the whole sentence instead of one by one to train. LSTM can decide what kinds of information
from previous words are passed to their successors.

Then, compared to LSTM, GRU is computationally cheaper and it can obtain the same result as LSTM, we try using both algorithms
to calculate the accuracy of Business category. The accuarcy of both algorithms are around 84%. 

After we search some paper, finally, we choose BiLSTM-Attention, which focusing limited attention on key information 
so that it can save resources and get the most effective information quickly.


Buliding Loss function:
We use CrossEntropyLoss() fuction from pytroch module. Because the calculation of CrossEntropyLoss() fuction can do both
log softmax and negative log likelihood loss, which combines them together. Besides, CrossEntropyLoss() fuction are
suitable for multi-calssfication problem because it is mainly used to determine how close the actual output is to the
expected output


convertNetOutput function：
In the evalution of hw2main.py, the output must be in the same format as the dataset ratings and business categories. So
we use torch.argmax() function which will Returns the indices of the maximum value of all elements in the input tensor.


Other parameters:
We have tried batch_size among 16,32,64 & 128 but found 32 most efficient and accurate, also 4 epchos is enough to train 
this model as with the epchos growth, the accuracy decreases on test case, which may indicates that the model will be overfitting. 

"""


import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
# import numpy as np
# import sklearn
import re
import torch.nn.functional as F

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """

    processed = sample.split(" ")

    return processed

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    result = []
    rule = re.compile("[^a-zA-Z\s\d]")
    for i in sample:
        i = rule.sub(' ', i)
        if (len(i) > 1):
            result.append(i)
    return result


def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch

#stopWords = {}
stopWords = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are",
             "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
             "can", "d", "did", "do", "does", "doing", "don", "down",
             "during", "each", "few", "for", "from", "further", "had", "hadn", "has",
             "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
             "if", "in", "into", "is",
             "it", "it's", "its", "itself", "just", "ll", "m",
             "ma", "me", "mightn", "more", "most", "my", "myself", "needn", "now", "o", "of",
             "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
             "over", "own", "re", "s", "same", "shan", "she", "she's", "should", "should've", "so", "some",
             "such", "t", "than", "that",
             "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these",
             "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very",
             "was", "we", "were", "what", "when", "where",
             "which", "while", "who", "whom", "why", "will", "with",
             "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
             "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's",
             "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's",
             "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's",
             "when's", "where's", "who's", "why's", "would"]

word_vector_switch = '6B100'
if word_vector_switch == '6B50':
    wordVectors = GloVe(name='6B', dim=50)
elif word_vector_switch == '6B100':
    wordVectors = GloVe(name = '6B',dim = 100)
elif word_vector_switch == '6B300':
    wordVectors = GloVe(name = '6B',dim = 300)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """
    encoded_tensor = torch.argmax(categoryOutput,dim = -1)
    ratingOutput = []
    categoryOutput = []
    for k in encoded_tensor:
        if k <= 4:
            ratingOutput.append(0)
            categoryOutput.append(k)
        else:
            ratingOutput.append(1)
            categoryOutput.append(k-5)
    
    ratingOutput = torch.tensor(ratingOutput).to(device)
    categoryOutput = torch.tensor(categoryOutput).to(device)

    return ratingOutput, categoryOutput

################################################################################
###################### The following determines the model ######################
################################################################################

#BiLSTM-Attention Model
#BiLSTM with attention involved
class BiLSTM_Attention(tnn.Module):
    def __init__(self):
        super(BiLSTM_Attention, self).__init__()
        self.input_size = 100
        self.hidden_size = 100
        self.is_bidirection = False
        self.lstm_layers = 1
        self.encoder = tnn.LSTM(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            num_layers = self.lstm_layers,
            bidirectional = True,
            batch_first = True,
            #dropout = 0.5,
        )
        self.fc1 = tnn.Linear(self.hidden_size*2,50)
        self.relu = tnn.ReLU()
        self.dropout = tnn.Dropout(0.5)
        self.decoder = tnn.Linear(50,10)
        
    def attention(self,output,final):
        hidden = final.view(-1, self.hidden_size * 2, self.lstm_layers)   # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attention_weights = torch.bmm(output,hidden).squeeze(2)
        soft_attention_weights = tnn.functional.softmax(attention_weights,1)
        soft_attention_weights = soft_attention_weights.unsqueeze(2)
        output = output.transpose(1,2)
        context = torch.bmm(output,soft_attention_weights).squeeze(2)
        return context

    def forward(self, input, length):
        #initialize weights
        h0 = torch.zeros(self.lstm_layers*2, len(input), self.hidden_size).to(device) # [num_layers * num_directions, batch_size, hidden_size]
        c0 = torch.zeros(self.lstm_layers*2, len(input), self.hidden_size).to(device) # [num_layers * num_directions, batch_size, hidden_size]

        output, (hidden, cell) = self.encoder(input, (h0, c0))#hidden = [layers*num_directions,batch_size,n_Hidden], output =  [batch_size, len_seq, n_hidden*2]
        output = self.attention(output, hidden)#updates weights by using attention model
        output = self.fc1(output)
        output = self.relu(output)
        output = self.decoder(output)
        return (0,output)


class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.loss = tnn.CrossEntropyLoss();


    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        result = self.convert_target(ratingTarget,categoryTarget).to(device)
        loss = self.loss(categoryOutput,result)
        return loss

    def convert_target(self,rt,ct):
        result = []

        #convert 10 classes into rating and categories
        for k in range(0,len(rt)):
            if rt[k] == 0:
                result.append(ct[k].item())
            else:
                result.append(5+ct[k].item())
        
        return torch.tensor(result)



net = BiLSTM_Attention()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize = 16
epochs = 4
#optimiser = toptim.SGD(net.parameters(), lr=0.01,momentum=0.9)
optimiser = toptim.Adam(net.parameters())
