In some ways this feels like this the first actual working implementation of the AUNN proposal. Here's what it has going fo it:

1. The only inputs are (positional encodings of) the index of a given token
2. The output is just the logit of the token
3. We can do autoregressive generation via backpropagating the net work on it's own predictions
4. The architecture is purely an MLP, no architecture-level considerations for modeling sequence dependencies

However, despite the surface level conformance to the original proposal I do unfortunately think this falls a bit flat of 'mission accomplished', at least in terms of embodying the original spirit of the AUNN proposal. Specifically, I have the following qualms:

1. The model requires bootstraping off of a (strangely architected) pre-existing LM. This isn't actually a criticism, it just dimishes the "organically emerges from scale" aspect a bit
2. Actually does have a somewhat explicit mechanism for tracking sequence state, in that the input and output embeddings to the LM basically serve the same purpose of a (tiny) RNN hidden state, which we are just memorizing on the fly.
3. Only works for autoregressive generation if we freeze the pretrained sequence model params
4. Autoregressive generation requires training in the output from the previous iteration into the output of an **early layer** in the next iteration, rather than just an end-to-end loss as tbe original seemed to suggest, and the later layers are just the original pretrained LM.