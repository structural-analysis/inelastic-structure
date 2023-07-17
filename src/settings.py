from pydantic import BaseSettings


class Settings(BaseSettings):
    example_name: str = "plate-square-inelastic-q8r"
    computational_zero = 1e-12
    isclose_tolerance = 1e-7
    examples_to_test = [
        "simple-beam-static-elastic",
        "simple-beam-static-inelastic",
        "torre-static-elastic",
        "skew-beams-static-elastic",
        "tripod-corner",
        "tripod-unload",
        "3story-static-elastic",
        "3story-static-inelastic-perfect",
        "3story-static-inelastic-softening",
        "torre-dynamic-elastic",
        "simple-beam-dynamic-elastic",
        "1story-dynamic-elastic",
        "2story-dynamic-elastic",
        "simple-beam-dynamic-inelastic-1phase",
        "1story-dynamic-inelastic-ll1.0-ap400k",
        "2story-dynamic-inelastic",
        "3d-2side-static-elastic",
        "3d-simple-beam-static-elastic",
        "3d-2side-dynamic-elastic",
        "3d-simple-beam-dynamic-elastic",
        "wall-1element-elastic",
        "wall-1element-elastic-q8r",
        "wall-9element-elastic",
        "wall-4element-elastic-q8r",
        "wall-1element-inelastic",
        "plate-1element-elastic-q8r",
        "plate-4element-elastic-q8r",
        "plate-9element-confined-elastic",
        "plate-confined-elastic",
        "plate-semiconfined-elastic",
        "plate-1element-inelastic-q8r",
    ]


settings = Settings()
