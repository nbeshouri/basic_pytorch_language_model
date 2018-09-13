# Basic PyTorch text generator

A basic (and heavily commented) word level RNN text generator created for pedagogical purposes.

The notebook implements a simple RNN language model that predicts that the next word given the previous word.

## Prerequisites

* Python 3.6 or higher
* `pytorch`
* `textblob`
* `joblib`
* `sklearn`

## Usage

Running the first two code cells sets up the needed classes and downloads the required data. From there, you can skip down and just load the pretained model, or you can play with the options and train your own. Options to play with:

* Replace the texts in the `data/texts` with texts of your own.
* The size of the recurrent layer.
* The number of recurrent layers.
* The type of recurrent layer (LSTM, GRU, RNN).
* The batch size.
* Different optimizers and learning rates.
* The length of the training sequences.

## Known issues

* There is no GPU support.
* I'm just predicting the most probable next word, which means that the model can get stuck in a loop repeating similar passages of text. A more complex approach would be to randomly sample the next word based on the predicted probabilities.
* Currently, each batch contains `batch_size` sequences of token ids. These sequences are randomly shuffled between batches, so we throw away the hidden state between batches because the ith sequence in batch 1 isn't contiguous with the ith sequence in batch 2. Ideally, we'd cut up the input text so that these were contiguous, and then the hidden state could be retained between batches, and the network could better learn to handle older hidden states (currently, it's only learning to deal with hidden states that have been used for 50 words, the length of the training sequences). Be careful here to detach the hidden states between batches (something like `hidden = hidden.detach()`) or it will try and backprop all the way back to the first input.