import cProfile
import pstats

from src.main import run

if __name__ == "__main__":
    # Run the profiler and save the results to a file
    profiler = cProfile.Profile()
    profiler.enable()

    # Call the function or code you want to profile
    run("wall-4element-inelastic-q8r")

    profiler.disable()
    profiler.dump_stats("profile_data.prof")
    profile_data = 'profile_data.prof'
    stats = pstats.Stats(profile_data)
    stats.strip_dirs().sort_stats('cumulative').print_stats(50)
