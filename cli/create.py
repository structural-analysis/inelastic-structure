import typer

from cli.functions.create_rectangular_plate import create_example

app = typer.Typer()


@app.command()
def hello(name: str):
    typer.echo(f"hello {name}")


@app.command()
def plate(name: str, xsize: float, ysize: float, xnum: int, ynum: int):
    create_example(name=name, xsize=xsize, ysize=ysize, xnum=xnum, ynum=ynum)


if __name__ == "__main__":
    app()
