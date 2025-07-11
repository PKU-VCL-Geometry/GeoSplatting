from __future__ import annotations

import subprocess
from typing import Optional


def run_command(cmd: str, *, verbose: bool = False) -> Optional[str]:
    """
    Runs a command and returns the output.

    Args:
        cmd: Command to run.
        verbose: If True, logs the output of the command.
    Returns:
        The output of the command if verbose is False, otherwise None.
    """
    out = subprocess.run(cmd, capture_output=not verbose, shell=True, check=False)
    if out.returncode != 0:
        if out.stderr:
            print(out.stderr.decode("utf-8"))
        raise RuntimeError(f"Error running command: {cmd}")
    if out.stdout is not None:
        return out.stdout.decode("utf-8")
    return None
