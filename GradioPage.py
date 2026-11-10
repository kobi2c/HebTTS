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
    Format,
    MbdMode,
    RandomMode,
    NikudMode,
    Speaker,
    SpeakerStore,
    CheckpointStore,
)
from make_sample import SampleMaker
from valle.data import AudioTokenizer, tokenize_audio
from valle.data.collation import get_text_token_collater
from valle.data.hebrew_root_tokenizer import AlefBERTRootTokenizer, replace_chars
from nikud_converter import convert_nikud_text, has_nikud


class GradioPage:
    def __init__(self, sampler: Optional[SampleMaker] = None, config: Optional[Config] = None) -> None:
        self.sampler            : SampleMaker                     = sampler or SampleMaker()
        self.config             : Config                          = (config or Config()).reset()
        self.checkpoint         : Optional[str]                   = None
        self.model              : Optional[torch.nn.Module]       = None
        self.mbd                : Optional[MbdMode]               = None
        self.audio_tokenizer    : Optional[AudioTokenizer]        = None
        self.text_collater      : Optional[Any]                   = None
        self.alef_bert_tokenizer: Optional[AlefBERTRootTokenizer] = None
        self.device             : torch.device                    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device_name        : str                             = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        self.demo               : Optional[gr.Blocks]             = None

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
        self.config.speaker = Speaker.create(
            mode=SpeakerMode(speaker_mode),
            id=str(speaker) if speaker is not None else None,
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
        if spk and spk.mode == SpeakerMode.Clone:
            spk_mode_str = str(SpeakerMode.Clone)
            spk_id = "osim"
            clone_aud = spk.resolve_audio_path() if (spk.exists or spk.data) else None
            clone_txt = spk.text or ""
        else:
            spk_mode_str = str(SpeakerMode.Predefined)
            spk_id = spk.id if spk else "osim"
            clone_aud = None
            clone_txt = ""

        return [
            self.config.checkpoint,
            spk_mode_str,
            spk_id,
            clone_aud,
            clone_txt,
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

    def load_model_if_needed(
        self, checkpoint_path: str, mbd: Union[MbdMode, int, bool] = MbdMode.Disabled
    ) -> Tuple[torch.nn.Module, AudioTokenizer, Any, AlefBERTRootTokenizer]:
        """Loads and caches the VALL-E model and tokenizers."""
        cp = CheckpointStore().find(checkpoint_path)
        actual_path = cp.full_path if cp else checkpoint_path

        if not os.path.exists(actual_path):
            if cp and not cp.exists and cp.gdrive_id:
                try:
                    actual_path = cp.download()
                except Exception as e:
                    raise gr.Error(f"Failed to download checkpoint '{checkpoint_path}': {e}")
            if not os.path.exists(actual_path):
                raise gr.Error(f"Checkpoint '{checkpoint_path}' not found on disk.")

        mbd_mode = MbdMode(int(mbd))

        if self.model is None or self.checkpoint != checkpoint_path:
            print(f"Loading checkpoint '{checkpoint_path}' on {self.device} ({self.device_name})...")
            t_model_start = time.time()
            if self.model is not None:
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.model, _ = self.sampler.load_model(str(actual_path), self.device)
            self.checkpoint = checkpoint_path
            model_dur = time.time() - t_model_start
            print(f"Model loaded successfully in {model_dur:.2f}s.")

        if self.audio_tokenizer is None or self.mbd != mbd_mode:
            print(f"Initializing AudioTokenizer (MBD={mbd_mode.name}) on {self.device}...")
            t_tok_start = time.time()
            self.audio_tokenizer = AudioTokenizer(mbd=bool(mbd_mode), device=self.device)
            self.mbd = mbd_mode
            tok_dur = time.time() - t_tok_start
            print(f"AudioTokenizer (MBD={mbd_mode.name}) initialized in {tok_dur:.2f}s.")

        if self.text_collater is None:
            self.text_collater = get_text_token_collater(self.sampler.tokens_file)

        if self.alef_bert_tokenizer is None:
            self.alef_bert_tokenizer = AlefBERTRootTokenizer(vocab_file=self.sampler.vocab_file)

        return self.model, self.audio_tokenizer, self.text_collater, self.alef_bert_tokenizer

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

        mbd_mode = MbdMode(int(cfg.mbd))
        random_mode = RandomMode(int(cfg.randomize_seed))
        nikud_mode = NikudMode(int(cfg.nikud_converter))

        # 1. Resolve speaker prompt audio and text
        spk = cfg.speaker
        if not spk:
            raise gr.Error("Speaker not configured.")
        if spk.mode == SpeakerMode.Predefined:
            prompt_audio = spk.resolve_audio_path()
            prompt_text = spk.text
        else:  # Custom Voice Clone
            if not spk.exists and not spk.data:
                raise gr.Error("Please upload a reference audio for voice cloning.")
            if not spk.text or not spk.text.strip():
                raise gr.Error("Please provide reference transcript text corresponding to the clone audio.")
            prompt_audio = spk.resolve_audio_path()
            prompt_text = spk.text.strip()

        # 2. Seed handling
        if random_mode == RandomMode.Enabled:
            seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
        else:
            seed = int(cfg.seed_number)
        torch.manual_seed(seed)

        # 3. Load model and tokenizers
        model, audio_tokenizer, text_collater, alef_bert_tokenizer = self.load_model_if_needed(cfg.checkpoint, mbd_mode)

        # 4. Tokenize reference audio prompt
        try:
            encoded_frames = tokenize_audio(audio_tokenizer, prompt_audio)
            audio_prompts = torch.concat([encoded_frames[0][0]], dim=-1).transpose(2, 1).to(self.device)
        except Exception as e:
            raise gr.Error(f"Failed to process reference audio: {str(e)}")

        # 5. Split input text into lines
        lines = [line.strip() for line in cfg.text.splitlines() if line.strip()]
        if not lines:
            raise gr.Error("No valid text lines found.")

        silence = torch.zeros((1, int(24000 * float(cfg.pause_duration))))
        segments: List[torch.Tensor] = []

        for i, line in enumerate(lines):
            progress((i, len(lines)), desc=f"Synthesizing line {i + 1}/{len(lines)}...")
            seg = self.sampler.synthesize_line(
                text=line,
                model=model,
                text_collater=text_collater,
                audio_tokenizer=audio_tokenizer,
                alef_bert_tokenizer=alef_bert_tokenizer,
                audio_prompts=audio_prompts,
                prompt_text=prompt_text,
                device=self.device,
                top_k=int(cfg.top_k),
                temperature=float(cfg.temperature),
                mbd=bool(mbd_mode),
                nikud_converter=bool(nikud_mode),
            )
            if seg.shape[-1] > 0:
                segments.append(seg)
                if i < len(lines) - 1 and cfg.pause_duration > 0:
                    segments.append(silence)

        if not segments:
            raise gr.Error("Failed to generate audio (empty tokens or invalid characters).")

        full_audio = torch.cat(segments, dim=-1)
        audio_np = full_audio.squeeze().cpu().numpy()

        max_val = np.abs(audio_np).max()
        if max_val > 1.0:
            audio_np = audio_np / max_val

        # Converted Ktiv-Male text if disambiguation was applied
        ktiv_male_text = ""
        if nikud_mode == NikudMode.Enabled and has_nikud(cfg.text):
            ktiv_male_text = convert_nikud_text(cfg.text)

        return (24000, audio_np), seed, ktiv_male_text

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
                        choices=["Predefined Speaker", "Custom Voice Clone"],
                        value=str(self.config.speaker.mode if self.config.speaker else SpeakerMode.Predefined),
                        label="Speaker Mode",
                    )

                    # Predefined speaker controls
                    with gr.Group(visible=True) as predefined_group:
                        speaker_dropdown = gr.Dropdown(
                            choices=speaker_names,
                            value=self.config.speaker.id if self.config.speaker else default_speaker,
                            label="Predefined Speaker",
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

            # Switch between Predefined Speaker and Custom Voice Clone groups
            def toggle_speaker_mode(mode: Union[SpeakerMode, int, str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
                is_predefined = (SpeakerMode(mode) == SpeakerMode.Predefined)
                return gr.update(visible=is_predefined), gr.update(visible=not is_predefined)

            speaker_mode_radio.change(
                fn=toggle_speaker_mode,
                inputs=[speaker_mode_radio],
                outputs=[predefined_group, clone_group],
            )

            # Update preview audio/text when predefined speaker changes
            def on_predefined_change(spk: str) -> Tuple[Optional[str], str]:
                aud, txt = get_speaker_preview(spk)
                return aud, txt

            speaker_dropdown.change(
                fn=on_predefined_change,
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
                    predefined_group,
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
