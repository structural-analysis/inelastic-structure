import cProfile
import pstats

from src.main import run

if __name__ == "__main__":
    # Run the profiler and save the results to a file
    profiler = cProfile.Profile()
    profiler.enable()

    # Call the function or code you want to profile
    run("3d-4story-3span-dynamic-inelastic")

    profiler.disable()
    profiler.dump_stats("profile_data.prof")
    profile_data = 'profile_data.prof'
    stats = pstats.Stats(profile_data)
    stats.strip_dirs().sort_stats('cumulative').print_stats(50)
