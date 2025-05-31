# %% [markdown]
# # Install and load required libraries

# %%
!pip install miditok
#!pip install symusic
#!pip install glob
#!pip install torch

# %%
import glob
import random
from typing import List
from collections import defaultdict

import numpy as np
from numpy.random import choice

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from symusic import Score
from miditok import REMI, TokenizerConfig

# %% [markdown]
# # Markov Chains for Next Chord Prediction
# (Ignore sections, bar lines, etc.)

# %%
a = open("data/chords.json")
dataset = []
for l in a.readlines():
    d = eval(l)
    dataset.append(d)

# %%
len(dataset)

# %%
dataset[0]

# %%
flatDataset = []
for d in dataset:
    flat = []
    for part in d['chords']:
        for bar in part:
            flat += bar
    flatDataset.append(flat)

# %%
flatDataset[0]

# %%
unigrams = defaultdict(int)
bigrams = defaultdict(int)

# %%
for d in flatDataset:
    for chord in d:
        unigrams[chord] += 1
    for (chord1,chord2) in zip(d[:-1],d[1:]):
        bigrams[(chord1,chord2)] += 1

# %%
unigramCounts = [(unigrams[k],k) for k in unigrams]
bigramCounts = [(bigrams[k],k) for k in bigrams]

# %%
unigramCounts.sort()
bigramCounts.sort()

# %%
unigramCounts[-10:]

# %%
bigramCounts[-10:]

# %%
dictionary = set(flatDataset[3])

# %% [markdown]
# ## Compute transition probabilities

# %%
transitions = defaultdict(list)
transitionProbabilities = defaultdict(list)

for b1,b2 in bigrams:
    if b1 in dictionary and b2 in dictionary:
        transitions[b1].append(b2)
        transitionProbabilities[b1].append(bigrams[(b1,b2)])

# %%
transitions

# %%
transitionProbabilities

# %%
def sample(length):
    seq = [random.choice(list(transitionProbabilities.keys()))]
    while len(seq) < length:
        probs = np.array(transitionProbabilities[seq[-1]])
        if not np.isclose(probs.sum(), 1.0):
            probs = probs / probs.sum()
        nextchord = choice(transitions[seq[-1]], 1, p=probs)
        seq.append(nextchord.item())
    return seq

# %%
chords = sample(10)
chords

# %% [markdown]
# ## Dump generated chord progressions

# %%
KEY_TO_IDX = {
    'C': 0,
    'C#': 1,
    'Db': 1,
    'D': 2,
    'D#': 3,
    'Eb': 3,
    'E': 4,
    'F': 5,
    'F#': 6,
    'Gb': 6,
    'G': 7,
    'G#': 8,
    'Ab': 8,
    'A': 9,
    'A#': 10,
    'Bb': 10,
    'B': 11,
    'Cb': 11,
}

# cover some of the qualities
QUALITY_TO_INTERVAL = {
    #        1     2     3  4     5     6     7
    '':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # major
    '-':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # minor
    '+':    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # augmented
    'o':    [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],  # diminished
    'sus':  [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # suspended
    '7':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],  # dominant 7th
    'j7':   [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],  # major 7th
    '-7':   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],  # minor 7th
    'o7':   [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],  # diminished 7th
    'm7b5': [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],  # half-diminished
    '6':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],  # major 6th
    '-6':   [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],  # minor 6th
    }

def chord_to_notes(chord):
    if len(chord) == 1 or chord[1] not in ['b','#']:
        root = chord[:1]
        quality = chord[1:]
    else:
        root = chord[:2]
        quality = chord[2:]

    bass = root

    root_c = 60
    bass_c = 36
    root_pc = KEY_TO_IDX[root]
    if quality not in QUALITY_TO_INTERVAL:
        raise ValueError('undefined chord quality {}'.format(quality))
    chord_map = list(np.where(np.array(QUALITY_TO_INTERVAL[quality]) == 1)[0])
    bass_pc = KEY_TO_IDX[bass]

    return [bass_c + bass_pc] + [root_c + root_pc + i for i in chord_map]

# %%
#pip install midiutil
from midiutil import MIDIFile # Import library

midi = MIDIFile(1) # Create a MIDI file that consists of 1 track
track = 0 # Set track number
time = 0 # Where is the event placed (at the beginning)
tempo = 120 # The tempo (beats per minute)
midi.addTempo(track, time, tempo) # Add tempo information

# %%
current_time = 0
default_duration = 4  # one beat
for chord in chords:
    notes = chord_to_notes(chord)
    for pitch in notes:
        midi.addNote(track, 0, pitch, current_time, default_duration, 100)
    current_time += default_duration

with open("chord_sample.mid", "wb") as f:
    midi.writeFile(f) # write MIDI file

# %% [markdown]
# # Markov Chain for MIDI generation
# ## Get the list of files for training and test sets

# %%
train_files = glob.glob("./data/train/*.mid")
test_files = glob.glob("./data/test/*.mid")

# %%
type(train_files[0])

# %%
train_files[0].encode('utf-8').decode('utf-8')

# %%
print(train_files[0].encode('utf-8'))

# %%
str.encode(train_files[0], 'utf-8')

# %% [markdown]
# ## Train your MIDI tokenizer

# %%
config = TokenizerConfig(num_velocities=1, use_chords=False, use_programs=True)
tokenizer = REMI(config)
tokenizer.train(vocab_size=1000, files_paths=train_files)
tokenizer.save("tokenizer.json")

# %% [markdown]
# ## Construct a PyTorch Dataset

# %%
class MIDIDataset(Dataset):
    def __init__(self, file_paths: List[str], tokenizer):
        self.tokenizer = tokenizer
        self.file_paths = file_paths
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        midi = Score(self.file_paths[idx])
        tokens = self.tokenizer(midi)
        return np.array(tokens)

# %% [markdown]
# ## Define PyTorch datasets and dataloaders

# %%
train_dataset = MIDIDataset(train_files, tokenizer)
test_dataset = MIDIDataset(test_files, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# %% [markdown]
# ## Define a Second Order Markov Chain model

# %%
class SecondOrderMarkovChain:
    def __init__(self):
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.probabilities = defaultdict(lambda: defaultdict(float))

    def train(self, train_loader):
        for sequence in train_loader:
            sequence = sequence[0].numpy().astype(int)
            for i in range(len(sequence) - 2):
                state1, state2 = sequence[i], sequence[i + 1]
                next_state = sequence[i + 2]
                self.transitions[(state1, state2)][next_state] += 1

        for (state1, state2), next_states in self.transitions.items():
            total = sum(next_states.values())
            for next_state, count in next_states.items():
                self.probabilities[(state1, state2)][next_state] = count / total
        return self.probabilities

    def generate(self, test_sequence, num_predictions=1):
        test_sequence = test_sequence[0].numpy().astype(int)
        results = [test_sequence[0], test_sequence[1]]
        for i in range(100):
            if (results[-2], results[-1]) not in self.probabilities:
                break
            else:
                probs = self.probabilities[(results[-2], results[-1])]
                states = list(probs.keys())
                probabilities = list(probs.values())
                if not states:
                    break
                try:
                    predictions = np.random.choice(states, size=num_predictions, p=probabilities)
                except:
                    break
                results.append(predictions[0])
        return results

# %% [markdown]
# ## Train your model and make inferences

# %%
model = SecondOrderMarkovChain()
model.train(train_loader)

predictions = []
for test_sequence in test_loader:
    predictions.append(model.generate(test_sequence))
for i, prediction in enumerate(predictions):
    output_score = tokenizer.decode(torch.Tensor(prediction))
    output_score.dump_midi(f"{i}.mid")

# %%
#!brew install wget
#!wget https://raw.githubusercontent.com/musescore/MuseScore/master/share/sound/FluidR3Mono_GM.sf3
#!pip install midi2audio
#!pip install IPython

# %%
from midi2audio import FluidSynth # Import library
from IPython.display import Audio, display
fs = FluidSynth("FluidR3Mono_GM.sf3") # Initialize FluidSynth
for i in range(len(predictions)):
    fs.midi_to_audio(f"{i}.mid", f"{i}.wav")
    display(Audio(f"{i}.wav"))

# %% [markdown]
# # RNN for MIDI generation

# %% [markdown]
# ## A New Dataset for batch inputs

# %%
from miditok.pytorch_data import DatasetMIDI, DataCollator

tokenizer = REMI()  # using defaults parameters (constants.py)
train_dataset = DatasetMIDI(
    files_paths=train_files,
    tokenizer=tokenizer,
    max_seq_len=1024,
    bos_token_id=tokenizer["BOS_None"],
    eos_token_id=tokenizer["EOS_None"],
)
test_dataset = DatasetMIDI(
    files_paths=test_files,
    tokenizer=tokenizer,
    max_seq_len=1024,
    bos_token_id=tokenizer["BOS_None"],
    eos_token_id=tokenizer["EOS_None"],
)
collator = DataCollator(tokenizer.pad_token_id)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collator)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collator)

# %%
len(train_loader), len(test_loader)

# %% [markdown]
# ## RNN

# %%
class MusicRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(MusicRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch_size, seq_length)
        x = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        out, hidden = self.rnn(x, hidden)  # out: (batch_size, seq_length, hidden_dim)
        out = self.fc(out)  # (batch_size, seq_length, vocab_size)
        return out, hidden

# %% [markdown]
# ### Training

# %%
def train(model, train_loader, val_loader, vocab_size, num_epochs=20, lr=0.001, device='cuda'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        # --------- Training ---------
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            batch = batch['input_ids'].to(device)  # (batch_size, seq_length)

            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            outputs = outputs.reshape(-1, vocab_size)
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --------- Validation ---------
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch['input_ids'].to(device)

                inputs = batch[:, :-1]
                targets = batch[:, 1:]

                outputs, _ = model(inputs)
                outputs = outputs.reshape(-1, vocab_size)
                targets = targets.reshape(-1)

                loss = criterion(outputs, targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")


# Example usage
if __name__ == "__main__":
    vocab_size = tokenizer.vocab_size
    embedding_dim = 256
    hidden_dim = 512
    num_layers = 2

    model = MusicRNN(vocab_size, embedding_dim, hidden_dim, num_layers)
    train(model, train_loader, test_loader, vocab_size)

# %% [markdown]
# ### Sampling

# %%
def sample(model, start_token, max_length=100, temperature=1.0, device='cuda'):
    model = model.to(device)
    model.eval()

    generated = [start_token]
    input_token = torch.tensor([[start_token]], device=device)  # (1, 1)

    hidden = None

    for _ in range(max_length):
        output, hidden = model(input_token, hidden)  # output: (1, 1, vocab_size)
        output = output[:, -1, :]  # take the last output
        output = output / temperature  # adjust randomness

        probs = F.softmax(output, dim=-1)  # (1, vocab_size)
        next_token = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_token)
        if next_token == 2 or next_token == 0: # reach end of sequence
          break

        input_token = torch.tensor([[next_token]], device=device)

    return generated

start_token = tokenizer.special_tokens_ids[1]
generated_sequence = sample(model, start_token, max_length=1024)

print("Generated token sequence:")
print(generated_sequence)

# %%
from midi2audio import FluidSynth # Import library
from IPython.display import Audio, display
fs = FluidSynth("FluidR3Mono_GM.sf3") # Initialize FluidSynth

output_score = tokenizer.tokens_to_midi([generated_sequence])
output_score.dump_midi(f"rnn.mid")
fs.midi_to_audio("rnn.mid", "rnn.wav")
display(Audio("rnn.wav"))

# %%



