from pydantic import BaseSettings


class Settings(BaseSettings):
    example_name: str = "simple_plate"
    computational_zero = 1e-10
    isclose_tolerance = 1e-7


settings = Settings()
