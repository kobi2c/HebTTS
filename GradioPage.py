import argparse
import base64
import json
import os
import sys
import tempfile
import time
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import utils
import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio
from config import (
    Config,
    SpeakerMode,
    SpeakerSample,
    Format,
    MbdMode,
    RandomMode,
    NikudMode,
    Speaker,
    LocalSpeaker,
    CloneSpeaker,
    SpeakerStore,
    CheckpointStore,
)
from engine import VoiceEngine, get_engine
from nikud_converter import convert_nikud_text, has_nikud


class GradioPage:
    def __init__(
        self,
        engine: Optional[Union[VoiceEngine, str]] = None,
        config: Optional[Config] = None,
    ) -> None:
        self.engine: VoiceEngine = get_engine(engine)
        self.config: Config = (config or Config()).reset()
        self.demo: Optional[gr.Blocks] = None

    @property
    def device(self) -> Any:
        return self.engine.device

    @property
    def device_name(self) -> str:
        return self.engine.device_name

    def update_config_from_ui(
        self,
        checkpoint: str,
        speaker_mode: Union[SpeakerMode, int, str],
        speaker: str,
        clone_audio: Optional[str],
        clone_text: str,
        text: str,
        pause_duration: float,
        temperature: float,
        top_k: int,
        mbd: Union[MbdMode, int, bool],
        seed_number: int,
        randomize_seed: Union[RandomMode, int, bool],
        nikud_converter: Union[NikudMode, int, bool],
        max_chunk_size: int = 100,
    ) -> Config:
        """Updates self.config attributes in-place from UI components with type conversions."""
        self.config.checkpoint = str(checkpoint)
        mode = SpeakerMode(speaker_mode)
        if mode == SpeakerMode.Local:
            self.config.speaker = LocalSpeaker.create(id=str(speaker) if speaker is not None else None)
        else:
            self.config.speaker = CloneSpeaker.create(
                clone_path=Path(clone_audio) if clone_audio else None,
                clone_text=clone_text,
            )
        self.config.text = str(text or "")
        self.config.pause_duration = float(pause_duration)
        self.config.temperature = float(temperature)
        self.config.top_k = int(top_k)
        self.config.mbd = MbdMode(int(mbd))
        self.config.seed_number = int(seed_number)
        self.config.randomize_seed = RandomMode(int(randomize_seed))
        self.config.nikud_converter = NikudMode(int(nikud_converter))
        self.config.max_chunk_size = int(max_chunk_size)
        return self.config

    def get_ui_values_from_config(self) -> List[Any]:
        """Returns list of values from self.config corresponding to the UI stateful components."""
        spk = self.config.speaker
        spk_mode_str = spk.ui_mode if spk else str(SpeakerMode.Local)
        sample = spk.ui_preview() if spk else SpeakerSample.Empty

        return [
            self.config.checkpoint,
            spk_mode_str,
            sample.id,
            sample.path or None,
            sample.text,
            self.config.text,
            self.config.pause_duration,
            self.config.temperature,
            self.config.top_k,
            int(self.config.mbd),
            self.config.seed_number,
            int(self.config.randomize_seed),
            int(self.config.nikud_converter),
            int(self.config.max_chunk_size),
        ]

    def export_settings(self, *args: Any) -> str:
        if len(args) == 1 and isinstance(args[0], Config):
            self.config = args[0]
        elif args:
            self.update_config_from_ui(*args)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as f:
            self.config.saveJson(f.name)
            return f.name

    def import_settings(self, file_path: Optional[str]) -> Tuple[Any, ...]:
        num_fields = len(self.get_ui_values_from_config())
        if not file_path:
            return tuple([gr.update()] * num_fields)
        try:
            self.config.loadJson(file_path)
            return tuple(self.get_ui_values_from_config())
        except Exception as e:
            print(f"Error loading settings: {e}")
            return tuple([gr.update()] * num_fields)

    def generate_audio(
        self,
        config: Optional[Config] = None,
        progress: gr.Progress = gr.Progress(track_tqdm=True),
    ) -> Tuple[Tuple[int, np.ndarray], int, str]:
        cfg = config or self.config
        if not isinstance(cfg, Config):
            raise TypeError(f"generate_audio accepts only a Config instance, got {type(cfg).__name__}")
        self.config = cfg

        if not cfg.text or not cfg.text.strip():
            raise gr.Error("Please provide Hebrew text to synthesize.")

        spk = cfg.speaker
        if not spk:
            raise gr.Error("Speaker not configured.")

        # Seed handling
        random_mode = RandomMode(int(cfg.randomize_seed))
        if random_mode == RandomMode.Enabled:
            seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
            cfg.seed_number = seed
        else:
            seed = int(cfg.seed_number)

        def report_progress(idx: int, tot: int, desc: str) -> None:
            progress((idx, tot), desc=desc)

        try:
            audio_np = self.engine.create(
                speaker=spk,
                text=cfg.text,
                config=cfg,
                progress_callback=report_progress,
            )
        except Exception as e:
            raise gr.Error(f"Speech synthesis failed: {str(e)}")

        if len(audio_np) == 0:
            raise gr.Error("Failed to generate audio (empty tokens or invalid characters).")

        # Converted Ktiv-Male text if disambiguation was applied
        ktiv_male_text = ""
        nikud_mode = NikudMode(int(cfg.nikud_converter))
        if nikud_mode == NikudMode.Enabled and has_nikud(cfg.text):
            ktiv_male_text = convert_nikud_text(cfg.text)

        return (self.engine.sample_rate, audio_np), seed, ktiv_male_text

    def build_interface(self) -> gr.Blocks:
        checkpoints = [c.path for c in CheckpointStore().list()]
        default_checkpoint = checkpoints[0] if checkpoints else "checkpoint.pt"

        speakers = SpeakerStore().list()
        speaker_names = [s.id for s in speakers]
        default_speaker = speakers[0].id if speakers else "osim"

        def get_speaker_preview(spk_id: str) -> Tuple[Optional[str], str]:
            spk = SpeakerStore().find(spk_id)
            if spk:
                return str(spk.full_path), spk.text
            return None, ""

        default_preview_audio, default_preview_text = get_speaker_preview(
            self.config.speaker.id if self.config.speaker else default_speaker
        )

        theme = gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        )

        with gr.Blocks(theme=theme, title="HebTTS - Hebrew Speech Synthesizer") as demo:
            gr.Markdown(
                f"""
                # 🎙️ HebTTS - Hebrew VALL-E Speech Synthesizer
                Synthesize natural Hebrew speech using **VALL-E** neural codec models and reference voice cloning.
                Running on **{self.device_name}** (`{self.device}`).
                """
            )

            with gr.Row():
                import_button = gr.UploadButton("📥 Import Settings", file_types=[".json"])
                export_button = gr.Button("📤 Export Settings")
                reset_button = gr.Button("🔄 Restore Defaults")
                export_file = gr.File(label="Exported Settings", visible=False)

            with gr.Row():
                # --- Left Column: Model & Speaker Configuration ---
                with gr.Column(scale=1):
                    gr.Markdown("### 1. Model & Voice")
                    checkpoint_dropdown = gr.Dropdown(
                        choices=checkpoints,
                        value=self.config.checkpoint,
                        label="Model Checkpoint",
                        info="Select VALL-E Hebrew checkpoint (.pt)",
                    )

                    speaker_mode_radio = gr.Radio(
                        choices=["Local Speaker", "Custom Voice Clone"],
                        value=self.config.speaker.ui_mode if self.config.speaker else str(SpeakerMode.Local),
                        label="Speaker Mode",
                    )

                    # Local speaker controls
                    with gr.Group(visible=True) as local_group:
                        speaker_dropdown = gr.Dropdown(
                            choices=speaker_names,
                            value=self.config.speaker.id if self.config.speaker else default_speaker,
                            label="Local Speaker",
                            info="Select a speaker from speakers/speakers.yaml",
                        )
                        speaker_preview_audio = gr.Audio(
                            value=default_preview_audio,
                            label="Speaker Prompt Audio",
                            type="filepath",
                            interactive=False,
                        )
                        speaker_preview_text = gr.Textbox(
                            value=default_preview_text,
                            label="Speaker Prompt Text",
                            lines=2,
                            interactive=False,
                        )

                    # Custom clone controls
                    with gr.Group(visible=False) as clone_group:
                        names_upper = " or ".join(m.name for m in Format)
                        exts_str = ", ".join(Format.extensions())
                        gr.Markdown(f"Upload reference audio (**24kHz mono {names_upper}**) and its transcript:")
                        clone_audio = gr.Audio(
                            label=f"Reference Voice Audio ({exts_str})",
                            type="filepath",
                        )
                        clone_text = gr.Textbox(
                            label="Reference Transcript Text",
                            lines=2,
                            placeholder="Type exact words spoken in reference audio...",
                        )
                        clone_txt_file = gr.File(
                            label="Or Upload Transcript File (.txt)",
                            file_types=[".txt"],
                        )

                # --- Right Column: Text & Generation ---
                with gr.Column(scale=1):
                    gr.Markdown("### 2. Hebrew Text")
                    text_input = gr.Textbox(
                        label="Text to Synthesize",
                        value=self.config.text,
                        lines=5,
                        placeholder="הזן טקסט בעברית לסינתזה...",
                    )
                    text_file_upload = gr.File(
                        label="Or Upload Input Text File (.txt)",
                        file_types=[".txt"],
                        height=110
                    )
                    ktiv_male_display = gr.Textbox(
                        label="Phonetic Ktiv-Male (Disambiguated)",
                        lines=3,
                        interactive=False,
                        show_copy_button=True,
                        placeholder="Phonetic Ktiv-Male text will appear here if Nikud disambiguation is applied...",
                        info="Read-only view showing vocalized words converted into disambiguated phonetic Ktiv-Male.",
                    )

                    generate_button = gr.Button("⚡ Generate Speech", variant="primary", size="lg")
                    output_audio = gr.Audio(
                        label="Generated Speech (24 kHz)",
                        type="numpy",
                        autoplay=True,
                    )

            # --- Advanced Parameters Accordion ---
            with gr.Accordion("⚙️ Advanced Parameters", open=False):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Conditioning & Vocoder")
                        pause_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.5,
                            value=self.config.pause_duration,
                            step=0.05,
                            label="Inter-sentence Pause (seconds)",
                            info="Silence duration inserted between lines.",
                        )
                        nikud_checkbox = gr.Checkbox(
                            label="Phonetic Nikud / Ktiv-Male Disambiguation",
                            value=bool(self.config.nikud_converter),
                            info="Use phonikud to phonetically convert vocalized words to Ktiv-Male.",
                        )
                        mbd_checkbox = gr.Checkbox(
                            label="Multi-Band Diffusion (MBD) Vocoder",
                            value=bool(self.config.mbd),
                            info="Enable MBD diffusion vocoder (if installed) instead of EnCodec decode.",
                        )
                        max_chunk_slider = gr.Slider(
                            minimum=20,
                            maximum=300,
                            value=self.config.max_chunk_size,
                            step=5,
                            label="Max Chunk Size (characters)",
                            info="Maximum character length before splitting long text into chunks.",
                        )

                    with gr.Column():
                        gr.Markdown("#### Sampling Controls")
                        with gr.Row():
                            temp_slider = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=self.config.temperature,
                                step=0.05,
                                label="Temperature",
                                info="Higher values = more creative/variable, lower = more deterministic.",
                            )
                            topk_slider = gr.Slider(
                                minimum=1,
                                maximum=100,
                                value=self.config.top_k,
                                step=1,
                                label="Top-K",
                                info="Top-K sampling parameter.",
                            )
                        with gr.Row():
                            seed_number = gr.Number(label="Seed", value=self.config.seed_number, precision=0)
                            randomize_seed = gr.Checkbox(label="Randomize Seed", value=bool(self.config.randomize_seed))

            # --- Interactive Callbacks ---

            # Switch between Local Speaker and Custom Voice Clone groups
            def toggle_speaker_mode(mode: Union[SpeakerMode, int, str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
                is_local = (SpeakerMode(mode) == SpeakerMode.Local)
                return gr.update(visible=is_local), gr.update(visible=not is_local)

            speaker_mode_radio.change(
                fn=toggle_speaker_mode,
                inputs=[speaker_mode_radio],
                outputs=[local_group, clone_group],
            )

            # Update preview audio/text when local speaker changes
            def on_local_change(spk: str) -> Tuple[Optional[str], str]:
                aud, txt = get_speaker_preview(spk)
                return aud, txt

            speaker_dropdown.change(
                fn=on_local_change,
                inputs=[speaker_dropdown],
                outputs=[speaker_preview_audio, speaker_preview_text],
            )

            # Populate clone text from uploaded .txt file
            def on_clone_txt_upload(file: Optional[Any]) -> Union[str, Dict[str, Any]]:
                if file is not None:
                    try:
                        filepath = file.name if hasattr(file, "name") else str(file)
                        with open(filepath, "r", encoding="utf-8") as f:
                            return f.read().strip()
                    except Exception as e:
                        return f"Error reading file: {e}"
                return gr.update()

            clone_txt_file.upload(
                fn=on_clone_txt_upload,
                inputs=[clone_txt_file],
                outputs=[clone_text],
            )

            # Populate text input and preview Ktiv-Male from uploaded text file
            def on_text_file_upload(file: Optional[Any], enabled: Union[NikudMode, int, bool]) -> Tuple[Union[str, Dict[str, Any]], str]:
                if file is not None:
                    try:
                        filepath = file.name if hasattr(file, "name") else str(file)
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                            km = convert_nikud_text(content) if (NikudMode(int(enabled)) == NikudMode.Enabled and has_nikud(content)) else ""
                            return content, km
                    except Exception as e:
                        return f"Error reading file: {e}", ""
                return gr.update(), gr.update()

            text_file_upload.upload(
                fn=on_text_file_upload,
                inputs=[text_file_upload, nikud_checkbox],
                outputs=[text_input, ktiv_male_display],
            )

            # Preview Ktiv-Male when text editing finishes or nikud toggle changes
            def on_text_or_nikud_change(text_val: str, enabled: Union[NikudMode, int, bool]) -> str:
                if NikudMode(int(enabled)) == NikudMode.Enabled and text_val and has_nikud(text_val):
                    return convert_nikud_text(text_val)
                return ""

            text_input.blur(
                fn=on_text_or_nikud_change,
                inputs=[text_input, nikud_checkbox],
                outputs=[ktiv_male_display],
            )

            nikud_checkbox.change(
                fn=on_text_or_nikud_change,
                inputs=[text_input, nikud_checkbox],
                outputs=[ktiv_male_display],
            )

            # Stateful Components (Order matches get_ui_values_from_config() / update_config_from_ui())
            stateful_components = [
                checkpoint_dropdown,
                speaker_mode_radio,
                speaker_dropdown,
                clone_audio,
                clone_text,
                text_input,
                pause_slider,
                temp_slider,
                topk_slider,
                mbd_checkbox,
                seed_number,
                randomize_seed,
                nikud_checkbox,
                max_chunk_slider,
            ]

            # Generate Audio (Accepts only Config instance)
            def on_ui_generate(*args: Any, progress: gr.Progress = gr.Progress(track_tqdm=True)) -> Tuple[Tuple[int, np.ndarray], int, str]:
                cfg = self.update_config_from_ui(*args)
                return self.generate_audio(cfg, progress=progress)

            generate_button.click(
                fn=on_ui_generate,
                inputs=stateful_components,
                outputs=[output_audio, seed_number, ktiv_male_display],
            )

            # Settings Import & Export
            export_button.click(
                fn=lambda *args: gr.update(value=self.export_settings(*args), visible=True),
                inputs=stateful_components,
                outputs=export_file,
            )

            import_button.upload(
                fn=self.import_settings,
                inputs=import_button,
                outputs=stateful_components,
            )

            # Restore Defaults Callback
            def reset_to_defaults():
                self.config.reset()
                spk_id = self.config.speaker.id if self.config.speaker else "osim"
                aud, txt = get_speaker_preview(spk_id)
                return (
                    *self.get_ui_values_from_config(),
                    aud,
                    txt,
                    gr.update(visible=True),
                    gr.update(visible=False),
                    "",
                )

            reset_button.click(
                fn=reset_to_defaults,
                inputs=[],
                outputs=[
                    *stateful_components,
                    speaker_preview_audio,
                    speaker_preview_text,
                    local_group,
                    clone_group,
                    ktiv_male_display,
                ],
            )

        self.demo = demo
        return demo

    def launch(self, host: str = "0.0.0.0", port: int = 7860, share: bool = False) -> Any:
        if self.demo is None:
            self.build_interface()
        return self.demo.launch(server_name=host, server_port=port, share=share)

    run = launch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HebTTS Gradio Web Interface")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    cli_args = parser.parse_args()

    page = GradioPage()
    share_env = os.getenv("GRADIO_SHARE", "False").lower() in ("true", "1", "t")
    page.launch(
        host=cli_args.host,
        port=cli_args.port,
        share=cli_args.share or share_env,
    )
