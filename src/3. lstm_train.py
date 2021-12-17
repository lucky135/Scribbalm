# -*- coding: utf-8 -*-
"""
Created on Fri May 21 14:42:55 2021

@author: Administrator
"""

import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def network():
    notes= get_notes()
    
    pitch= len(set(notes))
    
    train_input, train_output = get_sequences(notes, pitch)
     
    model = create_network(train_input, pitch)
     
    train(model,train_input,train_output)
     
     
def get_notes():
    
     notes = []
     for file in glob.glob("midi_files/*.mid"):
        midi = converter.parse(file)

        print("Parsing %s" % file)

        notes_to_parse = None

        try: 
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: 
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

        with open('data/notes', 'wb') as filepath:
            pickle.dump(notes, filepath)

     return notes

def get_sequences(notes, pitch):
    
    sequence_length = 100

    
    pitchnames = sorted(set(item for item in notes))

     
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    train_input = []
    train_output = []

    
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        train_input.append([note_to_int[char] for char in sequence_in])
        train_output.append(note_to_int[sequence_out])

    n_patterns = len(train_input)

    train_input = numpy.reshape(train_input, (n_patterns, sequence_length, 1))
    
    train_input = train_input / float(pitch)

    train_output = np_utils.to_categorical(train_output)

    return (train_input, train_output)

def create_network(train_input, pitch):

    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(train_input.shape[1], train_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(pitch))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

def train(model, train_input, train_output):
    
    filepath = "weights.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(train_input, train_output, epochs=20, batch_size=128, callbacks=callbacks_list)
    
if __name__ == '__main__':
    network()
