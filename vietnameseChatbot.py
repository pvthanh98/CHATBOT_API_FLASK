import nltk
import convert
from pyvi import ViTokenizer, ViPosTagger, ViUtils
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
#import pickle

with open("vi_2.json",'r', encoding='utf8') as file:
    data = json.load(file)
# try:
#     with open("data.pickle", "rb") as f:
#         words, labels, training, output = pickle.load(f)
# except:

words = [] #['hi','how','are','you',...]
labels = []
docs_x = [] #['hi','how are you',...]
docs_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        vietnamese_wrds = ViTokenizer.tokenize(pattern) # nhóm các từ ghép 
        wrds_no_accents = convert.convert(pattern) #loại bỏ dấu trong tiếng việt
        wrds_no_accents = nltk.word_tokenize(wrds_no_accents) #tách thành một mãng các từ không dấu
        vietnamese_wrds = nltk.word_tokenize(vietnamese_wrds) #tách thành mãng các từ có dấu
        wrds_accent_and_noaccent = vietnamese_wrds + wrds_no_accents #gộp tiếng việt có dấu và không dấu.
        words.extend(wrds_accent_and_noaccent) ###because use extend that use not append :::=> type(wrds):array -> so when append it will add [['hi'],['how','are','you']...]
        docs_x.append(wrds_accent_and_noaccent)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels: 
        labels.append(intent["tag"])

words = [w.lower() for w in words if w != "?"]
words = sorted(list(set(words))) #loai bo tu trung lap
labels = sorted(labels)
training = []
output = []

out_empty = [0 for _ in range(len(labels))]


for x, doc in enumerate(docs_x):
    bag = []
    wrds = [w.lower() for w in doc]
    print("words lap in doc",wrds)
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

    # with open("data.pickle", "wb") as f:
    #     pickle.dump((words,labels,training,output), f)

tensorflow.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

# try:
#model.load("model.tflearn")
# except:
model.fit(training, output, n_epoch=500, batch_size=8, show_metric=True)
model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s = s.lower()
    s_words = ViTokenizer.tokenize(s)
    s_words = nltk.word_tokenize(s_words)
    s_words = [word for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if se == w:
                bag[i] = 1

    return numpy.array(bag)

def chat():
    print("Start talking with th bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        result = model.predict([bag_of_words(inp,words)])[0]
        result_index = numpy.argmax(result)
        tag = labels[result_index]
        print("Tag:", tag)
        print(result)
        if result[result_index] > 0.5:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    response = tg["responses"]

            print(random.choice(response))
        else:
            print("Bot cannot understand you. Plz try gain@@@")

def message_response(sentence, charset='utf-8'):
    results = model.predict([bag_of_words(sentence, words)])[0]
    res_index = numpy.argmax(results)

    tag = labels[res_index]
    print("Tag:", tag)
    print(results)
    print(results[res_index]);
    if results[res_index] > 0.5:
        for tg in data["intents"]:
            if tg['tag'] == tag:
                response = tg['responses']
        return random.choice(response)

    else: 
        return "Bạn cứ nói tiếp chúng tôi đang lắng nghe"

    
#chat()





