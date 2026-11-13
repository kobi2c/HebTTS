from abc import ABC, abstractmethod
from typing import Any, List
import gradio as gr
from config import Config


class EnginePanel(ABC):
    """Abstract base class for engine-specific configuration panels in Gradio."""

    @abstractmethod
    def build_panel(self, config: Config) -> gr.Group:
        """Builds the Gradio component hierarchy inside a gr.Group and returns it."""
        ...

    @abstractmethod
    def get_components(self) -> List[gr.components.Component]:
        """Returns the list of stateful components in deterministic order."""
        ...

    @abstractmethod
    def extract_config(self, values: List[Any], config: Config) -> Config:
        """Updates and returns config from the list of UI values matching get_components()."""
        ...

    @abstractmethod
    def get_ui_values(self, config: Config) -> List[Any]:
        """Returns UI values from config in the order of get_components()."""
        ...


# Backward compatibility alias
EngineUI = EnginePanel

