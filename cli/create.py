import typer

from cli.functions.create_rectangular_plate import create_example

app = typer.Typer()


@app.command()
def salam():
    typer.echo("salam")


@app.command()
def plate(
    name: str,
    xsize: float,
    ysize: float,
    xnum: int,
    ynum: int,
    limit: int,
    t: float):

    create_example(
        name=name,
        xsize=xsize,
        ysize=ysize,
        xnum=xnum,
        ynum=ynum,
        limit=limit,
        t=t,
    )


if __name__ == "__main__":
    app()
