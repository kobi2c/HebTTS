from typing import Any, Dict, List, Optional
import gradio as gr

from config import Config, SeedMode
from ..EnginePanel import EnginePanel


EMOTION_NAMES: List[str] = [
    "Happiness",
    "Sadness",
    "Disgust",
    "Fear",
    "Surprise",
    "Anger",
    "Other",
    "Neutral",
]

EMOTION_PRESETS: Dict[str, List[float]] = {
    "Neutral / Automatic (Unconditional)": [0.0] * 8,
    "Happiness": [1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2],
    "Sadness":   [0.05, 1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2],
    "Disgust":   [0.05, 0.05, 1.0, 0.05, 0.05, 0.05, 0.05, 0.2],
    "Fear":      [0.05, 0.05, 0.05, 1.0, 0.05, 0.05, 0.05, 0.2],
    "Surprise":  [0.05, 0.05, 0.05, 0.05, 1.0, 0.05, 0.05, 0.2],
    "Anger":     [0.05, 0.05, 0.05, 0.05, 0.05, 1.0, 0.05, 0.2],
    "Other":     [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 1.0, 0.2],
    "Custom":    [0.0] * 8,
}


class ZonosPanel(EnginePanel):
    """Configuration panel for Zonos (Zyphra) continuous hybrid architecture engine."""

    def __init__(self) -> None:
        self.group: Optional[gr.Group] = None
        self.model_choice_dropdown: Optional[gr.Dropdown] = None
        self.language_dropdown: Optional[gr.Dropdown] = None
        self.cfg_scale_slider: Optional[gr.Slider] = None
        self.speaking_rate_slider: Optional[gr.Slider] = None
        self.pitch_std_slider: Optional[gr.Slider] = None
        self.fmax_slider: Optional[gr.Slider] = None
        self.dnsmos_slider: Optional[gr.Slider] = None
        self.vq_score_slider: Optional[gr.Slider] = None
        self.speaker_noised_checkbox: Optional[gr.Checkbox] = None
        self.emotion_preset_dropdown: Optional[gr.Dropdown] = None
        self.emotion_sliders: List[gr.Slider] = []
        self.min_p_slider: Optional[gr.Slider] = None
        self.top_p_slider: Optional[gr.Slider] = None
        self.top_k_slider: Optional[gr.Slider] = None
        self.linear_slider: Optional[gr.Slider] = None
        self.confidence_slider: Optional[gr.Slider] = None
        self.quadratic_slider: Optional[gr.Slider] = None
        self.seed_input: Optional[gr.Number] = None
        self.seed_mode_checkbox: Optional[gr.Checkbox] = None

    def build_panel(self, config: Config) -> gr.Group:
        with gr.Group(visible=(config.engine == "zonos")) as panel:
            self.group = panel
            gr.Markdown("### ⚡ Zonos (Zyphra) Architecture Parameters")
            with gr.Row():
                with gr.Column(scale=2):
                    self.model_choice_dropdown = gr.Dropdown(
                        choices=[
                            "notmax123/Zonos-Hebrew",
                            "Zyphra/Zonos-v0.1-transformer",
                            "Zyphra/Zonos-v0.1-hybrid",
                        ],
                        value=config.model_choice,
                        label="Model Variant",
                        info="Hebrew fine-tuned checkpoint or official Zyphra multilingual base model.",
                    )
                with gr.Column(scale=1):
                    self.language_dropdown = gr.Dropdown(
                        choices=["he", "en-us", "en-gb", "fr-fr", "de", "es", "cmn", "ja"],
                        value=config.language,
                        label="Language Code",
                        info="Conditioning language (default: 'he' for Hebrew).",
                    )
                    self.speaker_noised_checkbox = gr.Checkbox(
                        value=bool(config.speaker_noised),
                        label="Condition on Denoised Speaker",
                        info="Condition on cleaned reference audio.",
                    )

            gr.Markdown("#### 🎚️ Acoustic & Voice Conditioning")
            with gr.Row():
                self.cfg_scale_slider = gr.Slider(
                    minimum=1.0,
                    maximum=5.0,
                    value=float(config.cfg_scale),
                    step=0.1,
                    label="CFG Scale",
                    info="Classifier-Free Guidance strength (higher = clearer/stronger conditioning).",
                )
                self.speaking_rate_slider = gr.Slider(
                    minimum=5.0,
                    maximum=30.0,
                    value=float(config.speaking_rate),
                    step=0.5,
                    label="Speaking Rate",
                    info="Target pace of speech in phonemes/second.",
                )
                self.pitch_std_slider = gr.Slider(
                    minimum=10.0,
                    maximum=150.0,
                    value=float(config.pitch_std),
                    step=1.0,
                    label="Pitch Std Dev",
                    info="Intonation and pitch expressiveness.",
                )

            with gr.Row():
                self.fmax_slider = gr.Slider(
                    minimum=8000.0,
                    maximum=24000.0,
                    value=float(config.fmax),
                    step=500.0,
                    label="Fmax (Hz)",
                    info="High-frequency cutoff.",
                )
                self.dnsmos_slider = gr.Slider(
                    minimum=1.0,
                    maximum=5.0,
                    value=float(config.dnsmos_ovrl),
                    step=0.1,
                    label="DNSMOS Target Quality",
                    info="Deep Noise Suppression MOS quality target.",
                )
                self.vq_score_slider = gr.Slider(
                    minimum=0.5,
                    maximum=1.0,
                    value=float(config.vq_score),
                    step=0.01,
                    label="VQ Score",
                    info="Acoustic codebook preservation threshold.",
                )

            gr.Markdown("#### 🎭 Emotion Conditioning")
            emotions = list(config.emotions) if config.emotions else [0.0] * 8
            while len(emotions) < 8:
                emotions.append(0.0)

            # Determine initial preset matching config.emotions
            initial_preset = "Custom"
            for name, p_vals in EMOTION_PRESETS.items():
                if name != "Custom" and len(p_vals) == 8:
                    if all(abs(emotions[i] - p_vals[i]) < 1e-4 for i in range(8)):
                        initial_preset = name
                        break

            self.emotion_preset_dropdown = gr.Dropdown(
                choices=list(EMOTION_PRESETS.keys()),
                value=initial_preset,
                label="Emotion Preset",
                info="Select an emotion preset or expand sliders below to customize.",
            )

            with gr.Accordion("🎭 Fine-Grained Emotion Sliders (8 Dimensions)", open=False):
                self.emotion_sliders = []
                with gr.Row():
                    for i in range(4):
                        slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=float(emotions[i]),
                            step=0.05,
                            label=EMOTION_NAMES[i],
                        )
                        self.emotion_sliders.append(slider)
                with gr.Row():
                    for i in range(4, 8):
                        slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=float(emotions[i]),
                            step=0.05,
                            label=EMOTION_NAMES[i],
                        )
                        self.emotion_sliders.append(slider)

            with gr.Accordion("🎛️ Advanced Sampling Controls", open=False):
                with gr.Row():
                    self.min_p_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=float(config.min_p),
                        step=0.01,
                        label="Min-P",
                        info="Minimum probability threshold relative to top token (recommended: 0.1).",
                    )
                    self.top_p_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=float(config.top_p),
                        step=0.01,
                        label="Top-P",
                        info="Nucleus sampling threshold (0.0 to disable).",
                    )
                    self.top_k_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=int(config.top_k if config.top_k != 50 else 0),
                        step=1,
                        label="Top-K",
                        info="Vocabulary restriction (0 to disable).",
                    )
                with gr.Row():
                    self.linear_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=float(config.linear),
                        step=0.05,
                        label="NovelAI Linear",
                        info="Linear sampling modifier.",
                    )
                    self.confidence_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=float(config.confidence),
                        step=0.05,
                        label="NovelAI Confidence",
                        info="Confidence modifier.",
                    )
                    self.quadratic_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=float(config.quadratic),
                        step=0.05,
                        label="NovelAI Quadratic",
                        info="Quadratic modifier.",
                    )
                with gr.Row():
                    self.seed_input = gr.Number(label="Seed", value=config.seed, precision=0)
                    self.seed_mode_checkbox = gr.Checkbox(label="Randomize Seed", value=bool(config.seed_mode))

        return panel

    def get_components(self) -> List[gr.components.Component]:
        return [
            self.model_choice_dropdown,
            self.language_dropdown,
            self.cfg_scale_slider,
            self.speaking_rate_slider,
            self.pitch_std_slider,
            self.fmax_slider,
            self.dnsmos_slider,
            self.vq_score_slider,
            self.speaker_noised_checkbox,
            self.emotion_preset_dropdown,
            *self.emotion_sliders,
            self.min_p_slider,
            self.top_p_slider,
            self.top_k_slider,
            self.linear_slider,
            self.confidence_slider,
            self.quadratic_slider,
            self.seed_input,
            self.seed_mode_checkbox,
        ]

    def extract_config(self, values: List[Any], config: Config) -> Config:
        if not values or len(values) < 26:
            return config
        config.model_choice = str(values[0])
        config.language = str(values[1])
        config.cfg_scale = float(values[2])
        config.speaking_rate = float(values[3])
        config.pitch_std = float(values[4])
        config.fmax = float(values[5])
        config.dnsmos_ovrl = float(values[6])
        config.vq_score = float(values[7])
        config.speaker_noised = bool(values[8])
        # values[9] is emotion_preset_dropdown
        config.emotions = [float(x) for x in values[10:18]]
        config.min_p = float(values[18])
        config.top_p = float(values[19])
        config.top_k = int(values[20])
        config.linear = float(values[21])
        config.confidence = float(values[22])
        config.quadratic = float(values[23])
        config.seed = int(values[24])
        config.seed_mode = SeedMode(int(values[25]))
        return config

    def get_ui_values(self, config: Config) -> List[Any]:
        emotions = list(config.emotions) if config.emotions else [0.0] * 8
        while len(emotions) < 8:
            emotions.append(0.0)

        matched_preset = "Custom"
        for name, p_vals in EMOTION_PRESETS.items():
            if name != "Custom" and len(p_vals) == 8:
                if all(abs(emotions[i] - p_vals[i]) < 1e-4 for i in range(8)):
                    matched_preset = name
                    break

        return [
            config.model_choice,
            config.language,
            float(config.cfg_scale),
            float(config.speaking_rate),
            float(config.pitch_std),
            float(config.fmax),
            float(config.dnsmos_ovrl),
            float(config.vq_score),
            bool(config.speaker_noised),
            matched_preset,
            *[float(x) for x in emotions[:8]],
            float(config.min_p),
            float(config.top_p),
            int(config.top_k if config.top_k != 50 else 0),
            float(config.linear),
            float(config.confidence),
            float(config.quadratic),
            int(config.seed),
            int(config.seed_mode),
        ]

