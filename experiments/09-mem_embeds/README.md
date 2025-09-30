2025-09-28: The plan

Here's my intuition / hypothesis that underlies the rest of the plan:
Let's say you have some dataset composed of pairs of x,y
There exists some simple relationship between these pairs (ie y=x+eps, where eps is noise)
Let's say we first randomly shuffle all pairs of data within the dataset
Then, we assing each pair in the post shuffle data an absolute positional index within the dataset
We take a dense MLP and train it to predict: idx -> (x,y)
The hypothesis: if your dataset is N pairs long, the predictions at index N+1 will be some (random) values of 
x & y that obey the same relative relationship as the training datapoints. If you then backprop the next work only
on the relationship idx -> x_prime, where x_prine != x, then the value that net outputs for y will "snap" to a value 
that obeys the relationship with respect to x_prime.

Now, if the above happens to be true, then I believe we can use this to make a true end to end AUNN. We can do this by using an AUNN to distill wRNN states, there the tuple is (wRNN input @ i, wRNN output @ i).

It'll be interesting to see what the N+1 position output is without any backprop first

I suspect at the end of distillation training you haven't surpassed the functionality of the wRNN in any way

Question then is, after the initial distillation phase is there a continual learning scheme that allows for us to train the AUNN to surpass it's original wRNN source. The wRNN hidden state is very limited and so you'd expect relatively poor performance and long term coherence from just the distillation (original wRNN is only trained to be coherent for 64 steps). Can we train the AUNN such that we incentivize using the network weight updates to augment its own sequence prediction abilities?

A concrete test might be to initially train a wRNN to do a needle in haystack test, distil that wRNN, show that it can still do needle in haystack well for the sequence length you trained the original wRNN on, and then continually pretrain the wRNN and show that that it can do the haystack task at much longer sequence lengths.

some interesting ideas about transformer-xl / detached rnn states, gradient routing for cur loss vs next loss into 2 moes (or maybe 3 if including causal loss as an aux loss).