default:
    just --list

generate_sample:
    python generate_sample.py

train:
    python main.py

plot:
    python plot.py
