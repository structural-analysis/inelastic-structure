from pydantic import BaseSettings


class Settings(BaseSettings):
    example_name: str = "tripod"
    abar0: float = 0.77
    computational_zero = 1e-10


settings = Settings()
