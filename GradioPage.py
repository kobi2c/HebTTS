import argparse
import base64
import inspect
import json
import os
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from enum import IntEnum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, Union

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from config import (
    Config,
    SpeakerMode,
    SpeakerSample,
    Format,
    MbdMode,
    SeedMode,
    NikudMode,
    Speaker,
    LocalSpeaker,
    CloneSpeaker,
    SpeakerStore,
    CheckpointStore,
    NikudMaker,
    DEFAULT_TEXT,
    utils,
)
from engine import EnginePanel, EngineUI, VoiceEngine, get_engine, list_engines
from engine.heb_tts.HebTTSPanel import HebTTSPanel
from engine.zonos.ZonosPanel import EMOTION_NAMES, EMOTION_PRESETS, ZonosPanel


ENGINE_PANELS: Dict[str, Type[EnginePanel]] = {
    "hebtts": HebTTSPanel,
    "zonos": ZonosPanel,
}


def register_engine_panel(engine_id: str, panel_cls: Type[EnginePanel]) -> None:
    """Allows 3rd party modules or plugins to register custom engine panels."""
    ENGINE_PANELS[engine_id] = panel_cls


# =====================================================================
# Main GradioPage Application
# =====================================================================

class GradioPage:
    def __init__(
        self,
        engine: Optional[Union[VoiceEngine, str]] = None,
        config: Optional[Config] = None,
    ) -> None:
        self.config: Config = (config or Config()).reset()
        self._engines: Dict[str, VoiceEngine] = {}

        if engine is not None:
            if isinstance(engine, str):
                self.config.engine = engine
            else:
                eng_id = engine.info.id
                self.config.engine = eng_id
                self._engines[eng_id] = engine

        self.panels: Dict[str, EnginePanel] = {}
        for eng_id in list_engines():
            try:
                eng_inst = self._engines.get(eng_id) or get_engine(eng_id)
                self._engines[eng_id] = eng_inst
                self.panels[eng_id] = eng_inst.create_panel()
            except Exception:
                pass
        for eng_id, panel_cls in ENGINE_PANELS.items():
            if eng_id not in self.panels:
                self.panels[eng_id] = panel_cls()
        self.nikud_maker: NikudMaker = NikudMaker()
        self.demo: Optional[gr.Blocks] = None
        self.theme: Optional[Any] = None

    @property
    def engine(self) -> VoiceEngine:
        """Lazily retrieves or instantiates the active VoiceEngine."""
        eng_id = self.config.engine or "hebtts"
        if eng_id not in self._engines:
            self._engines[eng_id] = get_engine(eng_id)
        return self._engines[eng_id]

    @engine.setter
    def engine(self, eng: VoiceEngine) -> None:
        eng_id = eng.info.id
        self._engines[eng_id] = eng
        self.config.engine = eng_id

    @property
    def device(self) -> Any:
        return self.engine.device

    @property
    def device_name(self) -> str:
        return self.engine.device_name

    def get_speaker_preview(self, spk_id: Optional[str]) -> Tuple[Optional[str], str]:
        """Returns audio path and transcript text for a local speaker id."""
        if not spk_id:
            return None, ""
        spk = SpeakerStore().find(spk_id)
        if spk:
            return str(spk.full_path), spk.text
        return None, ""

    def get_engine_badge_markdown(self, engine_id: str) -> str:
        """Returns rich formatted badge markdown describing the active engine using engine metadata."""
        try:
            eng = self._engines.get(engine_id) or get_engine(engine_id)
            info = eng.live_info()
            return (
                f"⚡ **Active Engine**: `{info.name}` &nbsp;|&nbsp; "
                f"🔊 **Sample Rate**: `{info.sample_rate:,} Hz` &nbsp;|&nbsp; "
                f"🧠 **Architecture**: {info.description} &nbsp;|&nbsp; "
                f"💻 **Device**: `{info.device_name}` (`{info.device}`)"
            )
        except Exception:
            if engine_id == "zonos":
                return (
                    f"⚡ **Active Engine**: `Zonos (Zyphra)` &nbsp;|&nbsp; "
                    f"🔊 **Sample Rate**: `44,100 Hz` &nbsp;|&nbsp; "
                    f"🧠 **Architecture**: Continuous Hybrid Transformer & DAC Codec &nbsp;|&nbsp; "
                    f"💻 **Device**: `{self.device_name}` (`{self.device}`)"
                )
            return (
                f"⚡ **Active Engine**: `HebTTS (VALL-E)` &nbsp;|&nbsp; "
                f"🔊 **Sample Rate**: `24,000 Hz` &nbsp;|&nbsp; "
                f"🧠 **Architecture**: Neural Codec LM (VALL-E) & EnCodec &nbsp;|&nbsp; "
                f"💻 **Device**: `{self.device_name}` (`{self.device}`)"
            )

    def update_config_from_ui(self, *args: Any, **kwargs: Any) -> Config:
        """Updates self.config from UI component values or kwargs."""
        if args:
            engine_val = args[0]
            speaker_mode_val = args[1]
            speaker_val = args[2]
            clone_audio_val = args[3]
            clone_text_val = args[4]
            text_val = args[5]

            self.config.engine = str(engine_val)
            mode = SpeakerMode(speaker_mode_val)
            if mode == SpeakerMode.Local:
                self.config.speaker = LocalSpeaker.create(id=str(speaker_val) if speaker_val is not None else None)
            else:
                self.config.speaker = CloneSpeaker.create(
                    clone_path=Path(clone_audio_val) if clone_audio_val else None,
                    clone_text=str(clone_text_val or ""),
                )
            self.config.text = str(text_val or "")

            idx = 6
            hebtts_panel = self.panels.get("hebtts")
            if hebtts_panel:
                heb_len = len(hebtts_panel.get_components())
                heb_vals = list(args[idx:idx + heb_len])
                hebtts_panel.extract_config(heb_vals, self.config)
                idx += heb_len

            zonos_panel = self.panels.get("zonos")
            if zonos_panel:
                zon_len = len(zonos_panel.get_components())
                zon_vals = list(args[idx:idx + zon_len])
                zonos_panel.extract_config(zon_vals, self.config)
                idx += zon_len

        if kwargs:
            if "engine" in kwargs and kwargs["engine"] is not None:
                self.config.engine = str(kwargs["engine"])
            if any(k in kwargs for k in ("speaker_mode", "speaker", "clone_audio", "clone_text")):
                spk_mode = kwargs.get("speaker_mode", SpeakerMode.Local)
                mode = SpeakerMode(spk_mode)
                if mode == SpeakerMode.Local:
                    spk_id = kwargs.get("speaker", "osim")
                    self.config.speaker = LocalSpeaker.create(id=str(spk_id) if spk_id is not None else None)
                else:
                    ca = kwargs.get("clone_audio")
                    self.config.speaker = CloneSpeaker.create(
                        clone_path=Path(ca) if ca else None,
                        clone_text=str(kwargs.get("clone_text", "")),
                    )
            for k, v in kwargs.items():
                if k not in ("speaker_mode", "speaker", "clone_audio", "clone_text"):
                    if hasattr(self.config, k):
                        if k == "mbd":
                            setattr(self.config, k, MbdMode(int(v)))
                        elif k == "seed_mode":
                            setattr(self.config, k, SeedMode(int(v)))
                        elif k == "nikud_mode":
                            setattr(self.config, k, NikudMode(int(v)))
                        else:
                            setattr(self.config, k, v)

        return self.config

    def get_ui_values_from_config(self) -> List[Any]:
        """Returns list of values from self.config corresponding to all UI stateful components."""
        spk = self.config.speaker
        spk_mode_str = spk.ui_mode if spk else str(SpeakerMode.Local)
        sample = spk.ui_preview() if spk else SpeakerSample.Empty

        res: List[Any] = [
            self.config.engine,
            spk_mode_str,
            sample.id,
            sample.path or None,
            sample.text,
            self.config.text,
        ]

        if "hebtts" in self.panels:
            res.extend(self.panels["hebtts"].get_ui_values(self.config))
        if "zonos" in self.panels:
            res.extend(self.panels["zonos"].get_ui_values(self.config))

        return res

    def export_settings(self, *args: Any) -> str:
        """Saves current configuration to a JSON file and returns its path."""
        if len(args) == 1 and isinstance(args[0], Config):
            self.config = args[0]
        elif args:
            self.update_config_from_ui(*args)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as f:
            self.config.saveJson(f.name)
            return f.name

    def import_settings(self, file_path: Optional[str]) -> Tuple[Any, ...]:
        """Loads configuration from a JSON file and returns full state updates."""
        num_fields = len(self.get_ui_values_from_config())
        extra_count = 8
        if not file_path:
            return tuple([gr.update()] * (num_fields + extra_count))
        try:
            self.config.loadJson(file_path)
            spk_id = self.config.speaker.id if self.config.speaker else "osim"
            aud, txt = self.get_speaker_preview(spk_id)
            is_local = (self.config.speaker.ui_mode == str(SpeakerMode.Local)) if self.config.speaker else True
            is_heb = (self.config.engine == "hebtts")
            info_md = self.get_engine_badge_markdown(self.config.engine)
            audio_label = "Generated Speech (24 kHz)" if is_heb else "Generated Speech (44.1 kHz)"
            return (
                *self.get_ui_values_from_config(),
                aud,
                txt,
                gr.update(visible=is_local),
                gr.update(visible=not is_local),
                gr.update(visible=is_heb),
                gr.update(visible=not is_heb),
                info_md,
                gr.update(label=audio_label),
            )
        except Exception as e:
            print(f"Error loading settings: {e}")
            return tuple([gr.update()] * (num_fields + extra_count))

    def generate_audio(
        self,
        config: Optional[Config] = None,
        progress: gr.Progress = gr.Progress(track_tqdm=True),
    ) -> Tuple[Tuple[int, np.ndarray], int, str]:
        """Generates speech waveform using the active engine."""
        cfg = config or self.config
        if not isinstance(cfg, Config):
            raise TypeError(f"generate_audio accepts only a Config instance, got {type(cfg).__name__}")
        self.config = cfg

        if not cfg.text or not cfg.text.strip():
            raise gr.Error("Please provide Hebrew text to synthesize.")

        spk = cfg.speaker
        if not spk:
            raise gr.Error("Speaker not configured.")

        def report_progress(idx: int, tot: int, desc: str) -> None:
            progress((idx, tot), desc=desc)

        target_engine = self._engines.get(cfg.engine)
        if target_engine is None:
            target_engine = get_engine(cfg.engine)
            self._engines[cfg.engine] = target_engine

        try:
            audio_np = target_engine.create(
                speaker=spk,
                sample=spk.sample(),
                config=cfg,
                text=cfg.text,
                progress_callback=report_progress,
            )
        except Exception as e:
            raise gr.Error(f"Speech synthesis failed: {str(e)}")

        if len(audio_np) == 0:
            raise gr.Error("Failed to generate audio (empty tokens or invalid characters).")

        # Converted Ktiv-Male text if disambiguation was applied
        ktiv_male_text = ""
        nikud_mode = NikudMode(int(cfg.nikud_mode))
        if nikud_mode == NikudMode.Enabled and self.nikud_maker.has(cfg.text):
            ktiv_male_text = self.nikud_maker.convert(cfg.text)

        return (target_engine.info.sample_rate, audio_np), cfg.seed, ktiv_male_text

    def build_interface(self) -> gr.Blocks:
        """Constructs the tabbed, multi-engine Gradio interface."""
        speakers = SpeakerStore().list()
        speaker_names = [s.id for s in speakers]
        default_speaker = speakers[0].id if speakers else "osim"

        default_preview_audio, default_preview_text = self.get_speaker_preview(
            self.config.speaker.id if self.config.speaker else default_speaker
        )

        self.theme = gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        )

        blocks_kwargs: Dict[str, Any] = {"title": "HebTTS & Zonos - Hebrew Speech Studio"}
        if "theme" in inspect.signature(gr.Blocks.__init__).parameters:
            blocks_kwargs["theme"] = self.theme

        with gr.Blocks(**blocks_kwargs) as demo:
            gr.Markdown(
                f"""
                # 🎙️ Hebrew Speech Studio: HebTTS & Zonos
                Synthesize natural, highly expressive Hebrew speech using **VALL-E** neural codec models and **Zyphra Zonos** continuous hybrid architecture.
                """
            )

            # --- Top Toolbar: Settings Import/Export ---
            with gr.Row():
                import_button = gr.UploadButton("📥 Import Settings", file_types=[".json"])
                export_button = gr.Button("📤 Export Settings")
                reset_button = gr.Button("🔄 Restore Defaults")
                export_file = gr.File(label="Exported Settings", visible=False)

            # --- Tabbed Main Interface ---
            with gr.Tabs() as tabs:

                # =========================================================
                # TAB 1: Synthesis Playground
                # =========================================================
                with gr.Tab("🎙️ Synthesis Playground"):
                    with gr.Row():
                        # --- Left Column: Speaker & Voice Cloning ---
                        with gr.Column(scale=1):
                            gr.Markdown("### 1. Speaker & Reference Voice")
                            speaker_mode_radio = gr.Radio(
                                choices=["Local Speaker", "Custom Voice Clone"],
                                value=self.config.speaker.ui_mode if self.config.speaker else str(SpeakerMode.Local),
                                label="Speaker Mode",
                            )

                            # Local speaker group
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

                            # Custom clone group
                            with gr.Group(visible=False) as clone_group:
                                names_upper = " or ".join(m.name for m in Format)
                                exts_str = ", ".join(Format.extensions())
                                gr.Markdown(f"Upload reference voice audio (**{exts_str}**) and its transcript:")
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

                        # --- Right Column: Text & Audio Generation ---
                        with gr.Column(scale=1):
                            gr.Markdown("### 2. Hebrew Text & Audio Output")
                            text_input = gr.Textbox(
                                label="Text to Synthesize",
                                value=self.config.text,
                                lines=5,
                                placeholder="הזן טקסט בעברית לסינתזה...",
                            )
                            text_file_upload = gr.File(
                                label="Or Upload Input Text File (.txt)",
                                file_types=[".txt"],
                                height=110,
                            )
                            copy_kwargs: Dict[str, Any] = {}
                            textbox_params = inspect.signature(gr.Textbox.__init__).parameters
                            if "buttons" in textbox_params:
                                copy_kwargs["buttons"] = ["copy"]
                            elif "show_copy_button" in textbox_params:
                                copy_kwargs["show_copy_button"] = True

                            ktiv_male_display = gr.Textbox(
                                label="Phonetic Ktiv-Male (Disambiguated)",
                                lines=3,
                                interactive=False,
                                placeholder="Phonetic Ktiv-Male text will appear here if Nikud disambiguation is applied...",
                                info="Read-only view showing vocalized words converted into disambiguated phonetic Ktiv-Male.",
                                **copy_kwargs,
                            )

                            generate_button = gr.Button("⚡ Generate Speech", variant="primary", size="lg")
                            output_sr_label = "Generated Speech (24 kHz)" if self.config.engine == "hebtts" else "Generated Speech (44.1 kHz)"
                            output_audio = gr.Audio(
                                label=output_sr_label,
                                type="numpy",
                                autoplay=True,
                            )

                # =========================================================
                # TAB 2: Model & Engine Parameters
                # =========================================================
                with gr.Tab("⚙️ Model & Engine Parameters"):
                    gr.Markdown("### 🎯 Voice Engine Selection & Tuning")
                    with gr.Row():
                        engine_choices = [
                            ("HebTTS (VALL-E) - 24 kHz", "hebtts"),
                            ("Zonos (Zyphra) - 44.1 kHz", "zonos"),
                        ]
                        engine_dropdown = gr.Dropdown(
                            choices=engine_choices,
                            value=self.config.engine,
                            label="Active Voice Engine",
                            info="Select the speech synthesis neural architecture.",
                            scale=1,
                        )

                    engine_info_md = gr.Markdown(
                        self.get_engine_badge_markdown(self.config.engine)
                    )

                    # Build engine-specific parameter panels
                    hebtts_panel_group = self.panels["hebtts"].build_panel(self.config)
                    zonos_panel_group = self.panels["zonos"].build_panel(self.config)

            # --- Stateful Components Order ---
            stateful_components = [
                engine_dropdown,
                speaker_mode_radio,
                speaker_dropdown,
                clone_audio,
                clone_text,
                text_input,
                *self.panels["hebtts"].get_components(),
                *self.panels["zonos"].get_components(),
            ]

            # --- Interactive Callbacks ---

            # Engine selector change: toggle dynamic panels and badge
            def on_engine_change(engine_choice: str) -> Tuple[Dict[str, Any], Dict[str, Any], str, Dict[str, Any]]:
                is_heb = (engine_choice == "hebtts")
                info_md = self.get_engine_badge_markdown(engine_choice)
                audio_label = "Generated Speech (24 kHz)" if is_heb else "Generated Speech (44.1 kHz)"
                return (
                    gr.update(visible=is_heb),
                    gr.update(visible=not is_heb),
                    info_md,
                    gr.update(label=audio_label),
                )

            engine_dropdown.change(
                fn=on_engine_change,
                inputs=[engine_dropdown],
                outputs=[hebtts_panel_group, zonos_panel_group, engine_info_md, output_audio],
            )

            # Speaker Mode switch: toggle local vs clone groups
            def toggle_speaker_mode(mode: Union[SpeakerMode, int, str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
                is_local = (SpeakerMode(mode) == SpeakerMode.Local)
                return gr.update(visible=is_local), gr.update(visible=not is_local)

            speaker_mode_radio.change(
                fn=toggle_speaker_mode,
                inputs=[speaker_mode_radio],
                outputs=[local_group, clone_group],
            )

            # Local speaker dropdown change: update preview
            speaker_dropdown.change(
                fn=self.get_speaker_preview,
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
            hebtts_panel: HebTTSPanel = self.panels["hebtts"]  # type: ignore
            zonos_panel: ZonosPanel = self.panels["zonos"]      # type: ignore

            def on_text_file_upload(file: Optional[Any], enabled: Union[NikudMode, int, bool]) -> Tuple[Union[str, Dict[str, Any]], str]:
                if file is not None:
                    try:
                        filepath = file.name if hasattr(file, "name") else str(file)
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                            km = self.nikud_maker.convert(content) if (NikudMode(int(enabled)) == NikudMode.Enabled and self.nikud_maker.has(content)) else ""
                            return content, km
                    except Exception as e:
                        return f"Error reading file: {e}", ""
                return gr.update(), gr.update()

            text_file_upload.upload(
                fn=on_text_file_upload,
                inputs=[text_file_upload, hebtts_panel.nikud_checkbox],
                outputs=[text_input, ktiv_male_display],
            )

            # Preview Ktiv-Male on text blur or nikud checkbox change
            def on_text_or_nikud_change(text_val: str, enabled: Union[NikudMode, int, bool]) -> str:
                if NikudMode(int(enabled)) == NikudMode.Enabled and text_val and self.nikud_maker.has(text_val):
                    return self.nikud_maker.convert(text_val)
                return ""

            text_input.blur(
                fn=on_text_or_nikud_change,
                inputs=[text_input, hebtts_panel.nikud_checkbox],
                outputs=[ktiv_male_display],
            )

            hebtts_panel.nikud_checkbox.change(
                fn=on_text_or_nikud_change,
                inputs=[text_input, hebtts_panel.nikud_checkbox],
                outputs=[ktiv_male_display],
            )

            # Emotion Preset Selection Callback
            def on_preset_change(preset_name: str) -> Tuple[Any, ...]:
                if preset_name in EMOTION_PRESETS and preset_name != "Custom":
                    vals = EMOTION_PRESETS[preset_name]
                    return tuple(vals)
                return tuple([gr.update()] * 8)

            zonos_panel.emotion_preset_dropdown.change(
                fn=on_preset_change,
                inputs=[zonos_panel.emotion_preset_dropdown],
                outputs=zonos_panel.emotion_sliders,
            )

            # Emotion Slider Customization Callback
            def on_slider_user_input() -> str:
                return "Custom"

            for slider in zonos_panel.emotion_sliders:
                if hasattr(slider, "input"):
                    slider.input(
                        fn=on_slider_user_input,
                        inputs=[],
                        outputs=[zonos_panel.emotion_preset_dropdown],
                    )

            # Speech Generation Action
            def on_ui_generate(*args: Any, progress: gr.Progress = gr.Progress(track_tqdm=True)) -> Tuple[Tuple[int, np.ndarray], int, int, str]:
                cfg = self.update_config_from_ui(*args)
                (sr, audio_np), used_seed, ktiv_male = self.generate_audio(cfg, progress=progress)
                return (sr, audio_np), used_seed, used_seed, ktiv_male

            generate_button.click(
                fn=on_ui_generate,
                inputs=stateful_components,
                outputs=[output_audio, hebtts_panel.seed_input, zonos_panel.seed_input, ktiv_male_display],
            )

            # Settings Import & Export
            export_button.click(
                fn=lambda *args: gr.update(value=self.export_settings(*args), visible=True),
                inputs=stateful_components,
                outputs=export_file,
            )

            import_outputs = [
                *stateful_components,
                speaker_preview_audio,
                speaker_preview_text,
                local_group,
                clone_group,
                hebtts_panel_group,
                zonos_panel_group,
                engine_info_md,
                output_audio,
            ]

            import_button.upload(
                fn=self.import_settings,
                inputs=import_button,
                outputs=import_outputs,
            )

            # Restore Defaults
            def reset_to_defaults() -> Tuple[Any, ...]:
                self.config.reset()
                spk_id = self.config.speaker.id if self.config.speaker else default_speaker
                aud, txt = self.get_speaker_preview(spk_id)
                is_heb = (self.config.engine == "hebtts")
                info_md = self.get_engine_badge_markdown(self.config.engine)
                audio_label = "Generated Speech (24 kHz)" if is_heb else "Generated Speech (44.1 kHz)"
                return (
                    *self.get_ui_values_from_config(),
                    aud,
                    txt,
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(visible=is_heb),
                    gr.update(visible=not is_heb),
                    info_md,
                    gr.update(label=audio_label),
                    "",
                )

            reset_button.click(
                fn=reset_to_defaults,
                inputs=[],
                outputs=[
                    *import_outputs,
                    ktiv_male_display,
                ],
            )

        self.demo = demo
        return demo

    def launch(self, host: str = "0.0.0.0", port: int = 7860, share: bool = False, **kwargs: Any) -> Any:
        """Launches the Gradio web server application."""
        if self.demo is None:
            self.build_interface()
        launch_params = inspect.signature(self.demo.launch).parameters
        if "theme" in launch_params and "theme" not in kwargs and getattr(self, "theme", None) is not None:
            kwargs["theme"] = self.theme
        return self.demo.launch(server_name=host, server_port=port, share=share, **kwargs)

    run = launch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HebTTS & Zonos Gradio Web Interface")
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
