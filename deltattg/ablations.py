"""Shared typed description for parameterized DeltaTTG ablations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DeltaTTGAblation:
    method_name: str
    hydra_overrides: tuple[str, ...]
    meter_kind: str = "trained_ttg"

    def as_cli_args(self) -> list[str]:
        """Return a fresh list suitable for a Hydra launcher."""

        return list(self.hydra_overrides)



def load_ablation(name: str) -> DeltaTTGAblation:
    """Read a named variant from the installed root configs package."""
    from importlib.resources import files
    import yaml

    if not name or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in name):
        raise ValueError(f"Invalid ablation name: {name!r}")
    values = yaml.safe_load(files("configs.ablations").joinpath(name + ".yaml").read_text())
    values["hydra_overrides"] = tuple(values["hydra_overrides"])
    return DeltaTTGAblation(**values)
