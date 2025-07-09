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

demo_generate_sample:
    uv run sample_data/create_h5.py

demo_neuroglancer:
    uv run python -i sample_data/ng.py

demo:
    uv run demo.py
