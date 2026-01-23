import pstats
from typing import Annotated

import typer

app = typer.Typer()

@app.command()
def evalprofile(profile_to_eval: Annotated[str, typer.Option("--profile_to_eval", "-prof")]):
    '''
    Evaluate profiling stats from a given profile file.
    
    :param profile_to_eval: path to profile file generated with cProfile
    :type profile_to_eval: str

    Example usage on command line:
    uv run python -m cProfile -o profile_model.txt  model.py
    uv run profiling_stats.py  -prof profile_model.txt
    '''
    print(f"Evaluating profiling stats from: {profile_to_eval}")
    p = pstats.Stats(profile_to_eval)
    p.sort_stats('cumulative').print_stats(10)

  
if __name__ == "__main__":
    app()