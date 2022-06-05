from src.workshop import create_structure
from src.programming import solve_by_mahini_approach
from src.response import calculate_responses
from src.prepare import get_analysis_data

structure = create_structure()
analysis_data = get_analysis_data(structure)
result = solve_by_mahini_approach(analysis_data)
calculate_responses(structure, result)
