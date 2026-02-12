from rich.console import Console

_console = Console(highlight=False)
_err_console = Console(highlight=False, stderr=True)


def _prefix(worker: int = 0) -> str:
    if worker > 0:
        return f"[dim]\\[worker {worker}][/dim]"
    return "[bold blue]\\[yt-transcribe][/bold blue]"


def log(msg: str, worker: int = 0) -> None:
    _console.print(f"{_prefix(worker)} {msg}")


def log_step(step: str, msg: str, worker: int = 0) -> None:
    _console.print(f"{_prefix(worker)} [yellow]\\[{step}][/yellow] {msg}")


def log_done(msg: str, worker: int = 0) -> None:
    _console.print(f"{_prefix(worker)} [green]\\[done][/green] {msg}")


def log_error(msg: str, worker: int = 0) -> None:
    _err_console.print(f"{_prefix(worker)} [red]\\[error][/red] {msg}")
