# Symbolic Unconditioned Classical Piano Generation
This project was created by Charile Ngo and Samantha Phan on May/30 - June/3 for CSE 153: Machine Learning for Music. <br> <br>

Our task was symbolic uncondition music generation. We use techniques such as REMI tokenization, enhancing our LSTM model through additional layers, sampling technqiues such as top-p and top k and music functionality functions in order to promote music structure. <br>

## Dataset<br>
Why Maestro? We used Maestro due to realizing that classical music was better for musical training. It took trial and error, as we wanted to create edm music. This did not turn out well because of our limited computation resources in addition to pubic datasets avaliable.<br>

Our dataset was split into two data subsets, train and test in order to be used for our model.<br>

## Task1.ipynb/ Our Model for this Project <br>

Symbolic Tokenization/Data Processing

The MIDI files are converted to a REMI-inspired (REvamped MIDI) symbolic format designed to capture rich musical structure which includes:

* Note-On and Note-Duration tokens to represent pitch and rhythmic length
* Velocity tokens to capture the dynamics or loudness of a note
* Bar and Position tokens to encode time and beat placement
* Tempo tokens for tempo variation
* Chord tokens to represent harmonic structure


LTSM Model Architecture <br>

A multi-layer Long Short-Term Memory model was used as it captures long-term dependencies, being ideal for music generation.

- Embedding Layer: Converts token indices into dense vector embeddings of size d_model. This layer learns semantic relationships between different musical events.

- Stacked LSTM Layers: Two LSTM layers (optionally more) model sequential dependencies. Each LSTM layer has hidden_size units and includes dropout regularization to prevent overfitting.

- Linear Output Layer: Maps LSTM outputs to vocabulary-sized logits, which are passed through softmax for token sampling or classification.

- Dropout Layers: Applied after embeddings and between LSTM layers to improve generalization.

Training Setup<br>
 This model was trained using next-token prediction as our main objective was to minimize categorical cross-entropy between the predicted token distribution and the actual next token.

 + Input/Output Sequences: Fixed-length token windows are extracted from full sequences using a sliding window approach. Padding and masking ensure proper handling of variable-length sequences.

+ Optimizer: Adam optimizer is used with an initial learning rate scheduler for stability and faster convergence.

+ Batch Size: Typically 32 or 64 depending on available GPU memory.

+ Epochs: We did up to 50 epochs with early stopping based on validation loss in order to prevent overfitting.
  
+ Regularization: Dropout is applied at multiple points in the model to reduce overfitting.

Musicality Functions <br>

Sampling with Music Structure <br>

## Analysis
Basline.ipynb: <br>
- We used a basic Second-Order Markov and a basic LSTM to train our model in order to obtain a baseline we can compare too. <br>
- Audio Output:
  Markov:<br>
  LSTM: <br>

  As we can hear it sounds pretty awful and it is a good start to understand what not to strive for. This was used in our analysis against our model to prove that we have a "better" model.<br>

  Task1analysis.ipynb<br>

  
