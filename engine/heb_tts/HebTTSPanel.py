from typing import Any, List, Optional
import gradio as gr

from config import CheckpointStore, Config, MbdMode, NikudMode, SeedMode
from ..EnginePanel import EnginePanel


class HebTTSPanel(EnginePanel):
    """Configuration panel for HebTTS (VALL-E) neural codec engine."""

    def __init__(self) -> None:
        self.group: Optional[gr.Group] = None
        self.checkpoint_dropdown: Optional[gr.Dropdown] = None
        self.pause_slider: Optional[gr.Slider] = None
        self.temp_slider: Optional[gr.Slider] = None
        self.topk_slider: Optional[gr.Slider] = None
        self.mbd_checkbox: Optional[gr.Checkbox] = None
        self.seed_input: Optional[gr.Number] = None
        self.seed_mode_checkbox: Optional[gr.Checkbox] = None
        self.nikud_checkbox: Optional[gr.Checkbox] = None
        self.max_chunk_slider: Optional[gr.Slider] = None

    def build_panel(self, config: Config) -> gr.Group:
        checkpoints = [c.path for c in CheckpointStore().list()]
        default_checkpoint = checkpoints[0] if checkpoints else "checkpoint.pt"

        with gr.Group(visible=(config.engine == "hebtts")) as panel:
            self.group = panel
            gr.Markdown("### 🏛️ HebTTS (VALL-E) Architecture Parameters")
            with gr.Row():
                with gr.Column(scale=2):
                    self.checkpoint_dropdown = gr.Dropdown(
                        choices=checkpoints,
                        value=config.checkpoint or default_checkpoint,
                        label="Model Checkpoint",
                        info="Select Hebrew VALL-E neural codec checkpoint (.pt)",
                    )
                with gr.Column(scale=1):
                    self.mbd_checkbox = gr.Checkbox(
                        label="Multi-Band Diffusion (MBD) Vocoder",
                        value=bool(config.mbd),
                        info="Enable diffusion vocoder (if installed) instead of EnCodec decode.",
                    )
                    self.nikud_checkbox = gr.Checkbox(
                        label="Phonetic Nikud / Ktiv-Male Disambiguation",
                        value=bool(config.nikud_mode),
                        info="Use phonikud to phonetically convert vocalized words to Ktiv-Male.",
                    )

            gr.Markdown("#### ⏱️ Text Segmentation & Conditioning")
            with gr.Row():
                self.pause_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.5,
                    value=float(config.pause_duration),
                    step=0.05,
                    label="Inter-sentence Pause (seconds)",
                    info="Silence duration inserted between lines.",
                )
                self.max_chunk_slider = gr.Slider(
                    minimum=20,
                    maximum=300,
                    value=int(config.max_chunk_size),
                    step=5,
                    label="Max Chunk Size (characters)",
                    info="Maximum character length before splitting long text into chunks.",
                )

            gr.Markdown("#### 🎲 Autoregressive Sampling")
            with gr.Row():
                self.temp_slider = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=float(config.temperature),
                    step=0.05,
                    label="Temperature",
                    info="Higher values = more expressive/variable, lower = deterministic.",
                )
                self.topk_slider = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=int(config.top_k),
                    step=1,
                    label="Top-K",
                    info="Top-K sampling limit.",
                )
                self.seed_input = gr.Number(label="Seed", value=config.seed, precision=0)
                self.seed_mode_checkbox = gr.Checkbox(label="Randomize Seed", value=bool(config.seed_mode))

        return panel

    def get_components(self) -> List[gr.components.Component]:
        return [
            self.checkpoint_dropdown,
            self.pause_slider,
            self.temp_slider,
            self.topk_slider,
            self.mbd_checkbox,
            self.seed_input,
            self.seed_mode_checkbox,
            self.nikud_checkbox,
            self.max_chunk_slider,
        ]

    def extract_config(self, values: List[Any], config: Config) -> Config:
        if not values or len(values) < 9:
            return config
        (
            checkpoint,
            pause_duration,
            temperature,
            top_k,
            mbd,
            seed,
            seed_mode,
            nikud_mode,
            max_chunk_size,
        ) = values[:9]
        config.checkpoint = str(checkpoint)
        config.pause_duration = float(pause_duration)
        config.temperature = float(temperature)
        config.top_k = int(top_k)
        config.mbd = MbdMode(int(mbd))
        config.seed = int(seed)
        config.seed_mode = SeedMode(int(seed_mode))
        config.nikud_mode = NikudMode(int(nikud_mode))
        config.max_chunk_size = int(max_chunk_size)
        return config

    def get_ui_values(self, config: Config) -> List[Any]:
        return [
            config.checkpoint,
            float(config.pause_duration),
            float(config.temperature),
            int(config.top_k),
            int(config.mbd),
            int(config.seed),
            int(config.seed_mode),
            int(config.nikud_mode),
            int(config.max_chunk_size),
        ]

