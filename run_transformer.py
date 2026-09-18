import numpy as np, json

# ==============================  PROMPT =============================
prompt                        = ". the dog"

# ==============================  MODEL ==============================


w                             = {k: np.array(v) if k != "vocab" else v for k, v in json.load(open("weights.json")).items()}                  # here weights are being read in
vocab, d                      = w["vocab"], w["E"].shape[1]
words                         = prompt.split()
tokens                        = [vocab.index(t) for t in words]
n                             = len(tokens)

# 1. Embed: word meaning + position
x                             = w["E"][tokens] + w["Pos"][:n]

# 2. Self-attention: each word looks at itself and earlier words
Q, K, V                       = x @ w["Wq"].T, x @ w["Wk"].T, x @ w["Wv"].T
scores                        = Q @ K.T / np.sqrt(d)
scores[np.triu_indices(n, 1)] = -np.inf                                                                                                      # causal mask: a word cannot look ahead
softmax                       = lambda z: np.exp(z - z.max(-1, keepdims=True)) / np.exp(z - z.max(-1, keepdims=True)).sum(-1, keepdims=True) # define the softmax
A                             = softmax(scores)
x_att                         = x + (A @ V) @ w["Wo"].T                                                                                      # residual: attention output added to x

# 3. Feed-forward, applied to each position separately
hidden                        = np.maximum(0, x_att @ w["W1"].T + w["b1"])
x_ffn                         = x_att + hidden @ w["W2"].T + w["b2"]                                                                         # residual again

# 4. Score every vocabulary word against the last position's vector
logits                        = x_ffn[-1] @ w["E"].T
probs                         = softmax(logits)

# ============================== PRINT ==============================
def show(title, M, labels=None, cols=None):
    print("\n" + title)
    if cols: print(" " * 9 + "".join(f"{c:>7s}" for c in cols))
    for lab, row in zip(labels or range(len(np.atleast_2d(M))), np.atleast_2d(M)):
        print(f"{str(lab):>8s} " + "".join(f"{v:7.2f}" for v in row))

print("## MODEL WEIGHTS ##")
print("Embedding space: 4 dimensions")
show("Embedding weights E (rows = words, columns = dimensions):", w["E"], vocab)
show("Position weights Pos (rows = positions, columns = dimensions):", w["Pos"])
print("# Attention block #")
print("Query/Key space and value space: 4 dimensions, all equal for simplicity")
show("Query  weights Wq:", w["Wq"])
show("Key    weights Wk:", w["Wk"])
show("Value  weights Wv:", w["Wv"])
show("Output weights Wo:", w["Wo"])
print("# Perceptron layer #")
print("Rectified Linear Unit (ReLU), activation function is output=max(0,slope*input+intercept), W2*max(0,W1*x+b1)+b2")
print("Space defined by hidden perceptron layer has 8 dimensions")
show("FFN weights W1 (4 -> 8):", w["W1"])
show("FFN bias b1:", w["b1"])
show("FFN weights W2 (8 -> 4):", w["W2"])
show("FFN bias b2:", w["b2"])

print("## CALCULATIONS ##")
print("Prompt:", prompt, "-> token ids", tokens)
show("STEP 1 - x = E[tokens] + Pos (rows = prompt words):" , x    , words)
show("Query matrix Q = x @ Wq.T:"                          , Q    , words)
show("Key   matrix K = x @ Wk.T:"                          , K    , words)
show("Value matrix V = x @ Wv.T:"                          , V    , words)
show("Scores Q @ K.T / sqrt(d), masked (row = looking, column = looked at):", scores, words    , words)
show("Attention A = softmax(scores), each row sums to 1:"                   , A     , words    , words)
show("STEP 2 - x_att = x + (A @ V) @ Wo.T:"               , x_att , words)
show("Hidden layer after ReLU, max(0, x_att @ W1.T + b1):", hidden, words)
show("STEP 3 - x_ffn = x_att + hidden @ W2.T + b2:"       , x_ffn , words)
show("STEP 4 - logits = x_ffn[-1] @ E.T (one score per vocabulary word):"   , logits, ["logit"], vocab)
print("\nNext-word distribution p = softmax(logits):")
for word, p in sorted(zip(vocab, probs), key=lambda t: -t[1]):
    print(f"  {word:7s} {p:5.2f}  {'█' * int(40 * p)}")