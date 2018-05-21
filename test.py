# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from chatbot import clean_text, questionInt, test_predictions, answerInt2Word
######################## Testing #######################

# Loading the weights and Running the session
checkpoint = "./chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)

batch_size = 64

# Converting the questions from strings to lists of encoding integers
def convert_string2int(question, word2int):
    question = clean_text(question)
    return [word2int.get(word, word2int['<OUT>']) for word in question.split()]
 
# Setting up the chat
while(True):
    question = input("You: ")
    if question == 'Goodbye':
        break
    question = convert_string2int(question, questionInt)
    question = question + [questionInt['<PAD>']] * (25 - len(question))
    fake_batch = np.zeros((batch_size, 25))
    fake_batch[0] = question
    predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0]
    answer = ''
    for i in np.argmax(predicted_answer, 1):
        if answerInt2Word[i] == 'i':
            token = ' I'
        elif answerInt2Word[i] == '<EOS>':
            token = '.'
        elif answerInt2Word[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + answerInt2Word[i]
        answer += token
        if token == '.':
            break
    print('ChatBot: ' + answer)