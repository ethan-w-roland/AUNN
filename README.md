# Absolute Unit Neural Network (AUNN)

A simple implementation of Gwern's AUNN proposal:
https://gwern.net/aunn

## Background

First assign every token in a dataset a sequential index. Then, train a model to predict those tokens using nothing but the index. This is similar in concept to NeRFs, where the network is explicitly acting as a compressed representation of the dataset. The hope with AUNNs is that you could inference the network on unseen indices (i.e if the dataset is N tokens long, see what the prediction is for N+1) and that model would generalize well enought that the "next-index" prediction continues the sequence in a coherent way. You could imagine doing language modeling by training on one really long string of text (from The Pile, Wikipedia, etc), and asking the model what values are associated with the indices after the dataset ends.

In the proposal there's also the idea of "prompting" the model via conditioning-by-backpropogation. Essentially, you train the model via backprop that the "true" values for indices N+1, N+2, ..., N+P (where P is the length of your prompt) equates to the text of your prompt. You'd then see a logically coherent continuation of your prompt as the prediction for the post-prompt indices. At scale, this gets you behavior very similar to the "in-context-learning" you see in transformer or RNN models (or at least that's the hope).

I suggest Gwern's original article for more details. The architecture is very simple but the approach is a bit odd. Evidence of it working at all would be neat.

## Results

Intrigued by Gwern's proposal, I spent a few hours this week creating a simple implementation. The results were interesting enough to post. With a toy dataset, there's evidence both of generalization to unseen indices and of the ability to use backprop to "condition" on input for sequence-prediction tasks. The dataset has the following format:

`|bbb|aaa|aaa|aaa|aaa|ccc|ccc|bbb|aaa|bbb|ccc|bbb|bbb|bbb|aaa|bbb|ccc|....`

Every three-letter chunk is chosen randomly. In total I train on 100k randomly chosen sequences (aka 400k characters) for 500 epochs:

![loss graph](images/loss.png)

Once training is complete we reach 100% accuracy on the training set i.e. we've completely memorized the training set. Then we can query the model to get the predictions for N+1, N+2, ... , where N is the length of the dataset (400k chars) less one (for zero indexing). Doing this we get the following:

![output_N_N100](images/N-N100.png)

Although the model has never trained on indices > N, the model still generalizes to these unseen indices, in the sense that it preserves the local structure expected of the pattern ("|" every 4 chars, letters always come in sets of 3). It also generalizes to very to large input indices (greater than 10 million):

![output_N10million](images/N10million.png)

The value of a particular sequence in the training set is random, but the model still (very roughly) learns an even distribution of characters. I've made a histogram of all character predictions from N to N + 100k to illustrate the point:

![histogram](images/histogram.png)

I also try to demonstrate conditioning model outputs via a single step of backprop:

![conditioning](images/conditioning.png)

Here the original predictions for N+1 to N + 4 are `|aaa`.

I train the model with the values of `|c` for N+1 to N+2. 

Then, we see that N+3 to N+4 switch to `cc` in the post-conditioning output.

This shows (in a toy example at least) that the model can incorporate information provided within a single step of backprop while also utilizing the "knowledge" instilled in the model weights during pre-training. In this case the knowledge is the pattern that all letters must appear in triplets.

I found this little experiment interesting. Given more compute, maybe it could work for more complex language modeling tasks? 

Implementation-wise I'm just using a bog-standard MLP, with the Swish activation function, RMSNorm, and skip connections every 2 layers. The MLP has a hidden dimension of 64 and uses 8 layers. Much credit goes to GPT-o1 for helping me prototype. For the position embedding I'm using binary inputs (each dimension is a different digit of a 64 bit binary number). I also tried more standard fourier embeddings (as you might seen in LLM's) but surprisingly those seemed less performant, at least in the few tests I did. Though I doubt that holds true when modeling more complicated distributions.

![implementation](images/implementation.png)

The current version of this in the notebook includes some optimizations & misc changes I made after typing this original writeup. Time permitting, I'll try to do a few follow-ups. Simplest next step would be MNIST, and given that works I'll probably look into language modeling via TinyShakespeare.
