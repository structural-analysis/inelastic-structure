from enum import Enum
from pydantic import BaseSettings


class SiftingType(str, Enum):
    mahini = "mahini"
    not_used = None


class Settings(BaseSettings):
    example_name: str = "plate-semiconfined-inelastic"
    sifting_type: SiftingType = SiftingType.not_used
    computational_zero = 1e-12
    isclose_tolerance = 1e-7
    output_digits = 10
    examples_to_test = [
        # "3story-static-inelastic-softening",    # still not fixed after sifting
        # "3d-simple-beam-static-inelastic",  # possible bifurcation
        # "plate-semiconfined-inelastic",  # possible-bifurcation
        "simple-beam-static-elastic",
        "simple-beam-static-inelastic",
        "torre-static-elastic",
        "skew-beams-static-elastic",
        "tripod-corner",
        "tripod-unload",
        "3story-static-elastic",
        "3story-static-inelastic-perfect",
        "3d-2side-static-elastic",
        "3d-simple-beam-static-elastic",
        "3d-2side-static-inelastic",
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
        "plate-check-inelastic-q8r",
        "plate-square-inelastic-q8r",
        "wall-4element-inelastic-q8r",
        "plate-4element-inelastic-q8r",
        "plate-9element-confined-inelastic",
        "plate-confined-inelastic",
        # "torre-dynamic-elastic",
        # "simple-beam-dynamic-elastic",
        # "1story-dynamic-elastic",
        # "2story-dynamic-elastic",
        # "simple-beam-dynamic-inelastic-1phase",
        # "1story-dynamic-inelastic-ll1.0-ap400k",
        # "2story-dynamic-inelastic",
        # "3d-2side-dynamic-elastic",
        # "3d-simple-beam-dynamic-elastic",
        # "3d-simple-beam-dynamic-inelastic",
        # "3d-2side-dynamic-inelastic",
    ]


settings = Settings()
