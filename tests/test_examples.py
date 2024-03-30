from src.main import run
from src.settings import settings


for example_name in settings.examples_to_test:
    run(example_name)
