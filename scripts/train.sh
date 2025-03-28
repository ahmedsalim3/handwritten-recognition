#!/bin/bash

# Activate your virtual environment if needed
# source /path/to/your/virtualenv/bin/activate

python3 -m src.train_model --model cnn --epochs 20 --debug_mode
python3 -m src.train_model --model ffnn --epochs 20 --debug_mode