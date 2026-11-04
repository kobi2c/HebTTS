import argparse
import base64
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio

import utils  # Soundfile audio IO fallback and suppress torchvision warning
from sample import Sampler
from valle.data import AudioTokenizer, tokenize_audio
from valle.data.collation import get_text_token_collater
from valle.data.hebrew_root_tokenizer import AlefBERTRootTokenizer, replace_chars


class GradioPage:
    DEFAULT_TEXT: str = (
        "שלום, זהו מבחן דיבור עבור מודל הדיבור בעברית.\n"
        "המודל מאפשר שכפול קול ויצירת דיבור טבעי באיכות גבוהה."
    )

    SETTINGS_KEYS: List[str] = [
        "checkpoint",
        "speaker_mode",
        "predefined_speaker",
        "clone_audio",
        "clone_text",
        "text",
        "pause_duration",
        "temperature",
        "top_k",
        "mbd",
        "seed_number",
        "randomize_seed",
    ]

    def __init__(self, sampler: Optional[Sampler] = None) -> None:
        self.sampler            : Sampler                         = sampler or Sampler()
        self.checkpoint         : Optional[str]                   = None
        self.model              : Optional[torch.nn.Module]       = None
        self.mbd                : Optional[bool]                  = None
        self.audio_tokenizer    : Optional[AudioTokenizer]        = None
        self.text_collater      : Optional[Any]                   = None
        self.alef_bert_tokenizer: Optional[AlefBERTRootTokenizer] = None
        self.device             : torch.device                    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device_name        : str                             = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        self.demo               : Optional[gr.Blocks]             = None

    def get_available_checkpoints(self) -> List[str]:
        """Returns a list of available local checkpoints."""
        choices: List[str] = []
        for num, meta in self.sampler.CHECKPOINTS.items():
            if os.path.exists(meta["path"]):
                choices.append(meta["path"])
        # Also search current directory for any other .pt files
        for f in Path(".").glob("*.pt"):
            f_str = str(f)
            if f_str not in choices:
                choices.append(f_str)
        return choices or ["checkpoint.pt"]

    def get_default_settings(self) -> Dict[str, Any]:
        """Returns the default parameter settings dictionary."""
        checkpoints = self.get_available_checkpoints()
        default_checkpoint = checkpoints[0] if checkpoints else "checkpoint.pt"
        speakers_dict = self.sampler.get_speakers_dict()
        speaker_names = list(speakers_dict.keys())
        default_speaker = "osim" if "osim" in speaker_names else (speaker_names[0] if speaker_names else "")

        return {
            "checkpoint": default_checkpoint,
            "speaker_mode": "Predefined Speaker",
            "predefined_speaker": default_speaker,
            "clone_audio": None,
            "clone_text": "",
            "text": self.DEFAULT_TEXT,
            "pause_duration": 0.3,
            "temperature": 1.0,
            "top_k": 50,
            "mbd": False,
            "seed_number": 420,
            "randomize_seed": True,
        }

    def load_model_if_needed(
        self, checkpoint_path: str, mbd: bool
    ) -> Tuple[torch.nn.Module, AudioTokenizer, Any, AlefBERTRootTokenizer]:
        """Loads and caches the VALL-E model and tokenizers."""
        if not os.path.exists(checkpoint_path):
            raise gr.Error(f"Checkpoint '{checkpoint_path}' not found on disk.")

        if self.model is None or self.checkpoint != checkpoint_path:
            print(f"Loading checkpoint '{checkpoint_path}' on {self.device} ({self.device_name})...")
            if self.model is not None:
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.model, _ = self.sampler.load_model(checkpoint_path, self.device)
            self.checkpoint = checkpoint_path
            print("Model loaded successfully.")

        if self.audio_tokenizer is None or self.mbd != mbd:
            print(f"Initializing AudioTokenizer (MBD={mbd})...")
            self.audio_tokenizer = AudioTokenizer(mbd=mbd)
            self.mbd = mbd

        if self.text_collater is None:
            self.text_collater = get_text_token_collater(self.sampler.tokens_file)

        if self.alef_bert_tokenizer is None:
            self.alef_bert_tokenizer = AlefBERTRootTokenizer(vocab_file=self.sampler.vocab_file)

        return self.model, self.audio_tokenizer, self.text_collater, self.alef_bert_tokenizer

    def audio_to_base64(self, filepath: Optional[str]) -> Optional[Dict[str, str]]:
        if not filepath or not os.path.exists(filepath):
            return None
        with open(filepath, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(filepath)[1]
        return {"ext": ext, "data": encoded}

    def base64_to_audio(self, b64_dict: Optional[Dict[str, str]]) -> Optional[str]:
        if not b64_dict or "data" not in b64_dict:
            return None
        ext = b64_dict.get("ext", ".wav")
        decoded = base64.b64decode(b64_dict["data"])
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            f.write(decoded)
            return f.name

    def export_settings(self, *args: Any) -> str:
        settings: Dict[str, Any] = {}
        for key, value in zip(self.SETTINGS_KEYS, args):
            if key == "clone_audio":
                settings[key] = self.audio_to_base64(value)
            else:
                settings[key] = value

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)
            return f.name

    def import_settings(self, file_path: Optional[str]) -> Tuple[Any, ...]:
        if not file_path:
            return tuple([gr.update()] * len(self.SETTINGS_KEYS))
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                settings = json.load(f)
        except Exception as e:
            print(f"Error loading settings: {e}")
            return tuple([gr.update()] * len(self.SETTINGS_KEYS))

        outputs: List[Any] = []
        for key in self.SETTINGS_KEYS:
            if key not in settings:
                outputs.append(gr.update())
                continue
            val = settings[key]
            if key == "clone_audio":
                if val is not None and isinstance(val, dict):
                    outputs.append(self.base64_to_audio(val))
                else:
                    outputs.append(None)
            else:
                outputs.append(val)
        return tuple(outputs)

    def generate_audio(
        self,
        checkpoint: str,
        speaker_mode: str,
        predefined_speaker: str,
        clone_audio: Optional[str],
        clone_text: Optional[str],
        text: str,
        pause_duration: float,
        temperature: float,
        top_k: int,
        mbd: bool,
        seed_number: int,
        randomize_seed: bool,
        progress: gr.Progress = gr.Progress(track_tqdm=True),
    ) -> Tuple[Tuple[int, np.ndarray], int]:
        if not text or not text.strip():
            raise gr.Error("Please provide Hebrew text to synthesize.")

        # 1. Resolve speaker prompt audio and text
        speakers_dict = self.sampler.get_speakers_dict()

        if speaker_mode == "Predefined Speaker":
            if predefined_speaker not in speakers_dict:
                raise gr.Error(f"Predefined speaker '{predefined_speaker}' not found.")
            spk_data = speakers_dict[predefined_speaker]
            prompt_audio = str(Path(self.sampler.speaker_yaml).parent / spk_data["audio-prompt"])
            prompt_text = spk_data["text-prompt"]
        else:  # Custom Voice Clone
            if not clone_audio or not os.path.exists(clone_audio):
                raise gr.Error("Please upload a reference audio (.wav) for voice cloning.")
            if not clone_text or not clone_text.strip():
                raise gr.Error("Please provide reference transcript text corresponding to the clone audio.")
            prompt_audio = clone_audio
            prompt_text = clone_text.strip()

        # 2. Seed handling
        if randomize_seed:
            seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
        else:
            seed = int(seed_number)
        torch.manual_seed(seed)

        # 3. Load model and tokenizers
        model, audio_tokenizer, text_collater, alef_bert_tokenizer = self.load_model_if_needed(checkpoint, mbd)

        # 4. Tokenize reference audio prompt
        try:
            encoded_frames = tokenize_audio(audio_tokenizer, prompt_audio)
            audio_prompts = torch.concat([encoded_frames[0][0]], dim=-1).transpose(2, 1).to(self.device)
        except Exception as e:
            raise gr.Error(f"Failed to process reference audio: {str(e)}")

        # 5. Split input text into lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise gr.Error("No valid text lines found.")

        silence = torch.zeros((1, int(24000 * float(pause_duration))))
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
                top_k=int(top_k),
                temperature=float(temperature),
                mbd=mbd,
            )
            if seg.shape[-1] > 0:
                segments.append(seg)
                if i < len(lines) - 1 and pause_duration > 0:
                    segments.append(silence)

        if not segments:
            raise gr.Error("Failed to generate audio (empty tokens or invalid characters).")

        full_audio = torch.cat(segments, dim=-1)
        audio_np = full_audio.squeeze().cpu().numpy()

        max_val = np.abs(audio_np).max()
        if max_val > 1.0:
            audio_np = audio_np / max_val

        return (24000, audio_np), seed

    def build_interface(self) -> gr.Blocks:
        checkpoints = self.get_available_checkpoints()
        default_checkpoint = checkpoints[0] if checkpoints else "checkpoint.pt"

        speakers_dict = self.sampler.get_speakers_dict()
        speaker_names = list(speakers_dict.keys())
        default_speaker = "osim" if "osim" in speaker_names else (speaker_names[0] if speaker_names else "")

        def get_speaker_preview(spk_id: str) -> Tuple[Optional[str], str]:
            if spk_id in speakers_dict:
                spk_data = speakers_dict[spk_id]
                audio_p = str(Path(self.sampler.speaker_yaml).parent / spk_data["audio-prompt"])
                return audio_p, spk_data.get("text-prompt", "")
            return None, ""

        default_preview_audio, default_preview_text = get_speaker_preview(default_speaker)

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
                        value=default_checkpoint,
                        label="Model Checkpoint",
                        info="Select VALL-E Hebrew checkpoint (.pt)",
                    )

                    speaker_mode_radio = gr.Radio(
                        choices=["Predefined Speaker", "Custom Voice Clone"],
                        value="Predefined Speaker",
                        label="Speaker Mode",
                    )

                    # Predefined speaker controls
                    with gr.Group(visible=True) as predefined_group:
                        predefined_speaker = gr.Dropdown(
                            choices=speaker_names,
                            value=default_speaker,
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
                        gr.Markdown("Upload reference audio (**24kHz mono WAV**) and its transcript:")
                        clone_audio = gr.Audio(
                            label="Reference Voice Audio (.wav)",
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
                        value=self.DEFAULT_TEXT,
                        lines=5,
                        placeholder="הזן טקסט בעברית לסינתזה...",
                    )
                    text_file_upload = gr.File(
                        label="Or Upload Input Text File (.txt)",
                        file_types=[".txt"],
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
                            value=0.3,
                            step=0.05,
                            label="Inter-sentence Pause (seconds)",
                            info="Silence duration inserted between lines.",
                        )
                        mbd_checkbox = gr.Checkbox(
                            label="Multi-Band Diffusion (MBD) Vocoder",
                            value=False,
                            info="Enable MBD diffusion vocoder (if installed) instead of EnCodec decode.",
                        )

                    with gr.Column():
                        gr.Markdown("#### Sampling Controls")
                        with gr.Row():
                            temp_slider = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=1.0,
                                step=0.05,
                                label="Temperature",
                                info="Higher values = more creative/variable, lower = more deterministic.",
                            )
                            topk_slider = gr.Slider(
                                minimum=1,
                                maximum=100,
                                value=50,
                                step=1,
                                label="Top-K",
                                info="Top-K sampling parameter.",
                            )
                        with gr.Row():
                            seed_number = gr.Number(label="Seed", value=420, precision=0)
                            randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)

            # --- Interactive Callbacks ---

            # Switch between Predefined Speaker and Custom Voice Clone groups
            def toggle_speaker_mode(mode: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
                is_predefined = (mode == "Predefined Speaker")
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

            predefined_speaker.change(
                fn=on_predefined_change,
                inputs=[predefined_speaker],
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

            # Populate text input from uploaded text file
            def on_text_file_upload(file: Optional[Any]) -> Union[str, Dict[str, Any]]:
                if file is not None:
                    try:
                        filepath = file.name if hasattr(file, "name") else str(file)
                        with open(filepath, "r", encoding="utf-8") as f:
                            return f.read().strip()
                    except Exception as e:
                        return f"Error reading file: {e}"
                return gr.update()

            text_file_upload.upload(
                fn=on_text_file_upload,
                inputs=[text_file_upload],
                outputs=[text_input],
            )

            # Generate Audio
            generate_button.click(
                fn=self.generate_audio,
                inputs=[
                    checkpoint_dropdown,
                    speaker_mode_radio,
                    predefined_speaker,
                    clone_audio,
                    clone_text,
                    text_input,
                    pause_slider,
                    temp_slider,
                    topk_slider,
                    mbd_checkbox,
                    seed_number,
                    randomize_seed,
                ],
                outputs=[output_audio, seed_number],
            )

            # Settings Import & Export
            stateful_components = [
                checkpoint_dropdown,
                speaker_mode_radio,
                predefined_speaker,
                clone_audio,
                clone_text,
                text_input,
                pause_slider,
                temp_slider,
                topk_slider,
                mbd_checkbox,
                seed_number,
                randomize_seed,
            ]

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
                defaults = self.get_default_settings()
                aud, txt = get_speaker_preview(defaults["predefined_speaker"])
                return (
                    defaults["checkpoint"],
                    defaults["speaker_mode"],
                    defaults["predefined_speaker"],
                    defaults["clone_audio"],
                    defaults["clone_text"],
                    defaults["text"],
                    defaults["pause_duration"],
                    defaults["temperature"],
                    defaults["top_k"],
                    defaults["mbd"],
                    defaults["seed_number"],
                    defaults["randomize_seed"],
                    aud,
                    txt,
                    gr.update(visible=True),
                    gr.update(visible=False),
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
