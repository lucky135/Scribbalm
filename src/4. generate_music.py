# -*- coding: utf-8 -*-
"""
Created on Fri May 21 15:12:18 2021

@author: Administrator
"""

import pickle
import numpy
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import BatchNormalization as BatchNorm
from keras.layers import Activation

def generate():
    
     with open('data/notes', 'rb') as filepath:
        notes = pickle.load(filepath)
        
        pitchnames = sorted(set(item for item in notes))
        pitch= len(set(notes))
        
        test_input, normalized_input = get_sequences(notes, pitchnames, pitch)
        model = create_network(normalized_input, pitch)
        prediction_output = generate_notes(model, test_input, pitchnames, pitch)
        create_midi(prediction_output)
        
def get_sequences(notes, pitchnames, pitch):
    
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    test_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        test_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(test_input)

    
    normalized_input = numpy.reshape(test_input, (n_patterns, sequence_length, 1))
   
    normalized_input = normalized_input / float(pitch)

    return (test_input, normalized_input)

def create_network(test_input, pitch):
    
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(test_input.shape[1], test_input.shape[2]),
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


    model.load_weights('weights.hdf5')

    return model

def generate_notes(model, test_input, pitchnames, pitch):
   
    start = numpy.random.randint(0, len(test_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = test_input[start]
    prediction_output = []

    
    for note_index in range(500):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(pitch)

        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi(prediction_output):
    
    offset = 0
    output_notes = []

   
    for pattern in prediction_output:
        
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='test_output.mid')
    
if __name__ == '__main__':
    generate()
