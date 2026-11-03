from .cli import main

# Guarded: multiprocessing's spawn start method re-imports the main module in every worker
# (parallel insertion training, SubprocVecEnv).
if __name__ == "__main__":
    main()
