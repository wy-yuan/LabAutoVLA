"""Task language generation system for creating diverse natural language instructions.

This module provides utilities to generate natural language task descriptions
from templates and randomized parameters. Used during synthetic dataset generation
to create language variations for VLA training.

Example:
    Basic usage with a configuration dictionary::

        config = {
            "templates": [
                "Pipette {volume}µL from {source} to {target}",
                "Transfer {volume}µL from {source_adj} container to {target_adj}",
            ],
            "parameters": {
                "volume": {"type": "categorical", "values": [1, 5, 10, 50, 100]},
                "source": {"type": "categorical", "values": ["tube", "beaker"]},
                "target": {"type": "categorical", "values": ["plate", "dish"]},
                "source_adj": {"type": "categorical", "values": ["source", "reagent"]},
                "target_adj": {"type": "categorical", "values": ["destination", "target"]},
            },
        }

        gen = TaskLanguageTemplate("pipetting", config)
        prompt = gen.generate()  # "Pipette 50µL from beaker to plate"
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class TaskLanguageTemplate:
    """Generate natural language task descriptions from templates and parameters.

    This class manages:
    1. Template strings with placeholders for variable substitution
    2. Parameter definitions (ranges, discrete values, etc.)
    3. Random sampling of parameter values
    4. Template string formatting with sampled parameters

    Attributes:
        task_name: Name of the task (e.g., "pipetting")
        templates: List of template strings with {} placeholders
        parameter_ranges: Dict mapping parameter names to their configurations
        rng: NumPy random generator (created from seed if provided)
    """

    def __init__(
        self,
        task_name: str,
        config: dict[str, Any],
        seed: int | None = None,
    ):
        """Initialize language template system.

        Args:
            task_name: Human-readable name for the task
            config: Configuration dict with keys:
                - "templates": List[str] of template strings
                - "parameters": Dict[str, Dict] of parameter definitions
            seed: Random seed for reproducibility. If None, uses unseeded RNG.

        Raises:
            ValueError: If templates list is empty or parameters are invalid
        """
        self.task_name = task_name
        self.templates = config.get("templates", [])
        self.parameter_ranges = config.get("parameters", {})

        if not self.templates:
            log.warning(f"Task '{task_name}' has no templates. Generate will return empty string.")

        # Validate parameter definitions
        for param_name, param_cfg in self.parameter_ranges.items():
            param_type = param_cfg.get("type")
            if param_type not in ("categorical", "float", "int"):
                raise ValueError(f"Unknown parameter type '{param_type}' for '{param_name}'")

            if param_type == "categorical" and not param_cfg.get("values"):
                raise ValueError(f"Categorical parameter '{param_name}' must have 'values' list")

            if param_type in ("float", "int") and not param_cfg.get("range"):
                raise ValueError(f"Numeric parameter '{param_name}' must have 'range' tuple")

        # Initialize RNG
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def sample_parameters(self) -> dict[str, Any]:
        """Sample random values for all parameters.

        Sampling rules:
        - Categorical: uniform random choice from values list
        - Float: uniform random in range [min, max]
        - Int: uniform random integer in range [min, max]

        Returns:
            Dictionary mapping parameter names to sampled values
        """
        params = {}
        for param_name, param_cfg in self.parameter_ranges.items():
            param_type = param_cfg["type"]

            if param_type == "categorical":
                params[param_name] = self.rng.choice(param_cfg["values"])

            elif param_type == "float":
                min_val, max_val = param_cfg["range"]
                params[param_name] = self.rng.uniform(min_val, max_val)

            elif param_type == "int":
                min_val, max_val = param_cfg["range"]
                params[param_name] = self.rng.integers(min_val, max_val + 1)

        return params

    def generate(self, **kwargs) -> str:
        """Generate a task language instruction.

        If no kwargs provided, samples parameters randomly from the config.
        If kwargs provided, uses those values (for deterministic generation).

        Args:
            **kwargs: Optional explicit parameter values to override random sampling.
                      If not provided, parameters are sampled.

        Returns:
            Formatted language instruction string

        Raises:
            KeyError: If a template requires a parameter that wasn't provided
            ValueError: If templates list is empty
        """
        if not self.templates:
            log.warning(f"Task '{self.task_name}' has no templates. Returning empty string.")
            return ""

        # Sample parameters if not provided
        if not kwargs:
            params = self.sample_parameters()
        else:
            params = kwargs

        # Choose random template
        template = self.rng.choice(self.templates)

        # Format template with parameters
        try:
            instruction = template.format(**params)
        except KeyError as e:
            raise KeyError(
                f"Template '{template}' requires parameter {e.args[0]}, "
                f"but available parameters are: {list(params.keys())}"
            )

        return instruction

