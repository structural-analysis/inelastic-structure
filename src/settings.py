from pydantic import BaseSettings


class Settings(BaseSettings):
    example_name: str = "2story-dynamic-inelastic"
    computational_zero = 1e-10
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
    ]


settings = Settings()
