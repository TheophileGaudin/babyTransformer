# babyTransformer
**Transformers** are the key ingredient of Large Language Models (LLMs) which are driving the current AI boom.

This repo contains a 208-parameter transformer that is small enough to be fully transparent for self-learning and teaching purposes.
- `weights.json` contains the weights of the model
- `run_transformer.py` contains a prompt and running it predicts the probability distribution for the next word
- `train_transformer.py` is a script to train the model starting from random weights

To use:
1. Make sure Python is installed
2. The model is already trained. Running it only requires numpy and json which are here by default in typical Python installations. To run it, use `python run_transformer.py`
  
To train: 
1. You'll need the `torch` library. If you don't have it, run `pip install torch`.
2. Run `python train_transformer.py`
3. This takes a few seconds on a typical laptop CPU.

This is designed to be small and self-contained. Don't hesitate to **suggest improvements** by opening issues.
