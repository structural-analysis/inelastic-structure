from src.main import run

example_names = [
    # "1story_1span",
    "3story_perfect",
    "simple_beam",
    "skew_beams",
    # "triangle_truss",
    "tripod_corner",
    "tripod_unload",
]

for example_name in example_names:
    run(example_name)
