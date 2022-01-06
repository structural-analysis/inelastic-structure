from pydantic import BaseSettings


class Settings(BaseSettings):
    example_name: str = "simple_beam"


settings = Settings()
