#!/bin/bash

# python3 -m src.inference \
#     --model cnn \
#     --image_path "https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Number_1_in_green_rounded_square.svg/512px-Number_1_in_green_rounded_square.svg.png?20080406024426" \
#     --requests


python3 -m src.inference \
    --model ffnn \
    --image_path "https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Number_1_in_green_rounded_square.svg/512px-Number_1_in_green_rounded_square.svg.png?20080406024426" \
    --requests