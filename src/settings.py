from pydantic import BaseSettings


class Settings(BaseSettings):
    example_name: str = "simple_beam"
    load_limit: float = 280000


settings = Settings()
