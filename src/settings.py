from pydantic import BaseSettings


class Settings(BaseSettings):
    example_name: str = "simple_beam"
    abar0: float = 0.15


settings = Settings()
