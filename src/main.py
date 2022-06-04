from src.workshop import create_structure
from src.functions import prepare_raw_data
from src.programming import solve_by_mahini_approach
from src.response import calculate_responses

structure = create_structure()
mp_data = prepare_raw_data(structure=structure)
x_history = solve_by_mahini_approach(mp_data)
calculate_responses(structure, x_history, mp_data)
