default:
    just --list

generate_sample:
    uv run generate_sample.py

train:
    uv run main.py

plot:
    uv run plot.py

plot_local:
    python plot.py

demo:
    uv run demo.py
