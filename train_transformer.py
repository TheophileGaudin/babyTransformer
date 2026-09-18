import torch, json
torch.manual_seed(0)

# ---- 1. Corpus: 10-word vocabulary, "." marks both start and end ----
corpus       = ["the cat eats fish", "the dog eats meat", "a cat chases a dog", "the dog sees the cat", "a dog eats fish", "the cat sees a dog", "a cat eats meat", "the dog chases the cat"]
vocab        = sorted(set(" ".join(corpus).split()) | {"."}) #get vocabulary from corpus
idx          = {w: i for i, w in enumerate(vocab)}
data         = [torch.tensor([idx[w] for w in f". {s} .".split()]) for s in corpus]

# ---- 2. Parameters (V=10 words, T=7 positions, d=4 dims, h=8 hidden) ----
V, T, d, h   = len(vocab), 7, 4, 8
P            = lambda *s: torch.nn.Parameter(0.5 * torch.randn(*s))
W            = dict(E=P(V, d), Pos=P(T, d),                      # token & position embeddings
               Wq=P(d, d), Wk=P(d, d), Wv=P(d, d), Wo=P(d, d),   # attention
               W1=P(h, d), b1=P(h), W2=P(d, h), b2=P(d))         # feed-forward
               
# E has 10*4=40 parameters, Pos 7*4 parameters, Wq, Wk, Wv, Wo have 4*4 parameters, for a total of 4*4*4=64 parameters, W1 has 8*4=32 parameters, b1 8 parameters, W2 4*8=32 parameters, b2 4 parameters
# total: 40+28+64+32+8+32+4 = 208 parameters

def forward(tokens):                                                          # predicts the next token "logit" given a number of previous tokens ("forward pass" - what transformer.py does for a given prompt)
    n        = len(tokens)                                                    # number of tokens in the prompt
    x        = W["E"][tokens] + W["Pos"][:n]                                  # prompt embedding
    Q, K, Vv = x @ W["Wq"].T, x @ W["Wk"].T, x @ W["Wv"].T                    # query, key, value matrices for the prompt
    scores   = Q @ K.T / d**0.5                                               # logit scores for importance of every word for every other word
    mask     = torch.triu(torch.ones(n, n, dtype=bool), 1)                    # make sure we are causal (words don't know about words ahead)
    A        = torch.softmax(scores.masked_fill(mask, -1e9), -1)              # the attention matrix: how much each token (word) in the prompt attends to other tokens
    x        = x + (A @ Vv) @ W["Wo"].T                                       # embedding including the so-called "attention residual" (change due to the attention matrix)
    x        = x + torch.relu(x @ W["W1"].T + W["b1"]) @ W["W2"].T + W["b2"]  # applying Rectified Linear Unit (ReLU) activations in a perceptron layer (feed-forward)
    return     x @ W["E"].T                                                   # return the logits (roughly the "energy" of each possible word in the next position)

# ---- 3. Train: predict token t+1 from tokens 0..t ----
opt          = torch.optim.Adam(W.values(), lr=0.02)                          # Adam optimizer is a way to update weights that is stable and efficient
for step in range(1500):
    loss     = sum(torch.nn.functional.cross_entropy(forward(s[:-1]), s[1:]) for s in data) / len(data)    # calculates the loss at this step.
    opt.zero_grad()                                                                                        # clears previous step gradients (zero_grad).
    loss.backward()                                                                                        # calculates the new gradients via backpropagration (first we compute the loss gradients with respect to the logits dL/d(x @ W["E"].T), then we use the chain rule to calculate the gradients combining the old parameters at layer n-1 with gradients calculated at layer n. 
                                                                                                           # The chain rule applies as follows: dL/d(inputs of a layer) = dL/d(outputs of that layer) × d(outputs)/d(inputs), applied from the last layer backwards. The weights of each layer get their gradient from the same quantity, via d(outputs)/d(weights).
                                                                                                           # Very often, we have output = sum_i(parameter_i*input_i) d(output)/d(input_i) is simply parameter_i. So we often have dL/dinput_i = dL/doutput * parameter_i.
    opt.step()                                                                                             # This is calculating the step from the gradients, using something roughly equivalent to parameter_new = parameter_old - learning_rate*dL/doutput. As dL/doutput is the gradient, and we have a minus sign, we go against the gradient, so it's gradient descent (intent on descending the loss). Adam optimizer is slightly more sophisticated but the essence is this.
    if step % 300 == 0: print(f"step {step:4d}  loss {loss.item():.3f}")      # print loss every 300 steps
print(f"final loss {loss.item():.3f}")

# ---- 4. Save rounded weights ----
json.dump({"vocab": vocab, **{k: v.detach().round(decimals=2).tolist() for k, v in W.items()}}, open("weights.json", "w"), indent=1)
print("parameters:", sum(p.numel() for p in W.values()))