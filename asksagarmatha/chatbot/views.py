from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
import json,pickle
import numpy 
import nltk
import tflearn
from nltk.stem.lancaster import LancasterStemmer
import random
import tensorflow
from chatbot.models import Question

def load_model(training,output):
    tensorflow.reset_default_graph()
    input_layer = tflearn.input_data(shape=[None, len(training[0])]) #46
    dense1 = tflearn.fully_connected(input_layer, 8)
    softmax = tflearn.fully_connected(dense1, len(output[0]), activation="softmax")
    net = tflearn.regression(softmax)
    model = tflearn.DNN(net)
    model.load("/home/sanjog/asksagarmatha/demomodel2.tflearn")
    print(len(training[0]),len(output[1]))
    return model


def home(request):
    return render(request,'chatbot/index.html')
    
def message(request):
    if request.method=='POST' and request.headers.get('x-requested-with') == 'XMLHttpRequest':
        msg=request.POST.get('msg')
        print(msg)
        rs=chat(msg)
        data = {
            'msg':rs
        }
        print(msg,rs)
        q=Question(question_text=msg,reply=data['msg'])
        q.save()

        return JsonResponse(data)
    return JsonResponse({'msg':'can not get'})

def bag_of_words(s, words):
    with open("/home/sanjog/asksagarmatha/stopwords.pickle", "rb") as f:
        stopwords =pickle.load(f)
    
    stemmer = LancasterStemmer()
    bag = []
    wrds=nltk.word_tokenize(s)
    wrds=[word.lower() for word in wrds if word.isalpha()]
    wrds = [word for word in wrds if word not in stopwords ]
    wrds=[stemmer.stem(word)for word in wrds ]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)            
            
            
    return numpy.array(bag)




def chat(msg):
    with open("/home/sanjog/asksagarmatha/chatbot.json") as file:
        data = json.load(file)
   
    with open("/home/sanjog/asksagarmatha/data2.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
    model=load_model(training,output)
    results = model.predict([bag_of_words(msg, words)])
    print(results)
    results_index = numpy.argmax(results)
    if results[0][results_index] < 0.5:
        return "sorry i didn't understand. can you be more specific"
    
    tag = labels[results_index]

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
            return random.choice(responses)

        
# Create your views here.

