from pydantic import BaseSettings


class Settings(BaseSettings):
    example_name: str = "triangle_truss"
    computational_zero = 1e-10


settings = Settings()
