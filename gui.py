"""A small desktop GUI for running NeuTTS locally.

The GUI intentionally uses only the Python standard-library (tkinter) so it can
open before project dependencies are installed. Use the "Install dependencies"
button once after downloading the repository, then generate speech from the same
window. The default install intentionally installs the published NeuTTS wheel,
which bundles espeak-ng, while avoiding editable-package CMake builds on Windows.
"""

from __future__ import annotations

import importlib
import os
import queue
import subprocess
import sys
import threading
import traceback
import webbrowser
from pathlib import Path
from tkinter import BOTH, DISABLED, END, LEFT, NORMAL, RIGHT, filedialog, messagebox, ttk
import tkinter as tk

APP_DIR = Path(__file__).resolve().parent
DEFAULT_TEXT = (
    "My name is Andy. I'm 25 and I just moved to London. The underground is "
    "pretty confusing, but it gets me around in no time at all."
)
BACKBONES = [
    "neuphonic/neutts-nano",
    "neuphonic/neutts-air",
    "neuphonic/neutts-nano-q4-gguf",
    "neuphonic/neutts-air-q4-gguf",
    "neuphonic/neutts-nano-french",
    "neuphonic/neutts-nano-german",
    "neuphonic/neutts-nano-spanish",
]
VOICE_ROLES = {
    "Jo（默认女声）": ("samples/jo.wav", "samples/jo.txt"),
    "Dave（男声）": ("samples/dave.wav", "samples/dave.txt"),
    "Greta（女声）": ("samples/greta.wav", "samples/greta.txt"),
    "Juliette（女声）": ("samples/juliette.wav", "samples/juliette.txt"),
    "Mateo（男声）": ("samples/mateo.wav", "samples/mateo.txt"),
    "自定义": ("", ""),
}
DEFAULT_VOICE_ROLE = "Jo（默认女声）"
DEFAULT_SPEED = 0
DEFAULT_VOLUME = 0
DEFAULT_PITCH = 0
CORE_DEPENDENCIES = ["neutts==1.2.0"]
UNINSTALL_PACKAGES = [
    "neutts",
    "neucodec",
    "llama-cpp-python",
    "onnxruntime",
    "resemble-perth",
    "phonemizer",
]


class NeuTTSGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("NeuTTS 一键语音生成工具")
        self.geometry("1040x760")
        self.minsize(920, 660)
        self.log_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self._build_style()
        self._build_ui()
        self.after(100, self._drain_log_queue)

    def _build_style(self) -> None:
        self.configure(bg="#101827")
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#101827")
        style.configure("Card.TFrame", background="#172033", relief="flat")
        style.configure("TLabel", background="#101827", foreground="#e5e7eb", font=("Segoe UI", 10))
        style.configure("Card.TLabel", background="#172033", foreground="#e5e7eb", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 22, "bold"), foreground="#f8fafc", background="#101827")
        style.configure("Hint.TLabel", foreground="#9ca3af", background="#172033")
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=8)
        style.configure("TButton", font=("Segoe UI", 10), padding=7)
        style.configure("TEntry", fieldbackground="#0f172a", foreground="#f8fafc", insertcolor="#f8fafc")
        style.configure("TCombobox", fieldbackground="#0f172a", foreground="#f8fafc")
        style.configure("Horizontal.TProgressbar", troughcolor="#0f172a", background="#38bdf8")

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=20)
        root.pack(fill=BOTH, expand=True)

        header = ttk.Frame(root)
        header.pack(fill="x", pady=(0, 16))
        ttk.Label(header, text="NeuTTS GUI", style="Title.TLabel").pack(side=LEFT)
        ttk.Button(header, text="打开项目主页", command=lambda: webbrowser.open("https://github.com/neuphonic/neutts")).pack(side=RIGHT)

        main = ttk.PanedWindow(root, orient="horizontal")
        main.pack(fill=BOTH, expand=True)

        left = ttk.Frame(main, style="Card.TFrame", padding=16)
        right = ttk.Frame(main, style="Card.TFrame", padding=16)
        main.add(left, weight=3)
        main.add(right, weight=2)

        self._build_generate_card(left)
        self._build_install_card(right)
        self._build_log_card(right)

    def _build_generate_card(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="文本转语音", style="Card.TLabel", font=("Segoe UI", 15, "bold")).pack(anchor="w")
        ttk.Label(parent, text="首次运行会自动从 HuggingFace 下载模型，请保持网络畅通。", style="Hint.TLabel").pack(anchor="w", pady=(2, 12))

        ttk.Label(parent, text="要朗读的文本", style="Card.TLabel").pack(anchor="w")
        self.text_input = tk.Text(parent, height=8, wrap="word", bg="#0f172a", fg="#f8fafc", insertbackground="#f8fafc", relief="flat", padx=10, pady=10, font=("Segoe UI", 10))
        self.text_input.insert("1.0", DEFAULT_TEXT)
        self.text_input.pack(fill="x", pady=(5, 12))

        self.ref_audio = tk.StringVar(value=str(APP_DIR / "samples" / "jo.wav"))
        self.ref_text = tk.StringVar(value=str(APP_DIR / "samples" / "jo.txt"))
        self.output_path = tk.StringVar(value=str(APP_DIR / "output.wav"))
        self.voice_role = tk.StringVar(value=DEFAULT_VOICE_ROLE)
        self.speed = tk.IntVar(value=DEFAULT_SPEED)
        self.volume = tk.IntVar(value=DEFAULT_VOLUME)
        self.pitch = tk.IntVar(value=DEFAULT_PITCH)
        self.backbone = tk.StringVar(value=BACKBONES[0])
        self.device = tk.StringVar(value="cpu")

        role_row = ttk.Frame(parent, style="Card.TFrame")
        role_row.pack(fill="x", pady=5)
        ttk.Label(role_row, text="语音角色", style="Card.TLabel", width=14).pack(side=LEFT)
        role_box = ttk.Combobox(role_row, textvariable=self.voice_role, values=list(VOICE_ROLES), state="readonly")
        role_box.pack(side=LEFT, fill="x", expand=True, padx=8)
        role_box.bind("<<ComboboxSelected>>", self._apply_voice_role)
        ttk.Button(role_row, text="应用", command=self._apply_voice_role).pack(side=LEFT)

        self._path_row(parent, "参考音频 .wav", self.ref_audio, [("WAV files", "*.wav"), ("All files", "*.*")])
        self._path_row(parent, "参考文本 .txt", self.ref_text, [("Text files", "*.txt"), ("All files", "*.*")])
        self._path_row(parent, "输出文件", self.output_path, [("WAV files", "*.wav")], save=True)

        grid = ttk.Frame(parent, style="Card.TFrame")
        grid.pack(fill="x", pady=(4, 12))
        ttk.Label(grid, text="模型", style="Card.TLabel").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Combobox(grid, textvariable=self.backbone, values=BACKBONES).grid(row=0, column=1, sticky="ew", padx=(10, 0))
        ttk.Label(grid, text="设备", style="Card.TLabel").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Combobox(grid, textvariable=self.device, values=["cpu", "cuda", "mps", "gpu"], state="readonly").grid(row=1, column=1, sticky="ew", padx=(10, 0))
        grid.columnconfigure(1, weight=1)

        self._build_parameter_card(parent)

        buttons = ttk.Frame(parent, style="Card.TFrame")
        buttons.pack(fill="x")
        self.generate_button = ttk.Button(buttons, text="开始生成语音", style="Accent.TButton", command=self.generate_audio)
        self.generate_button.pack(side=LEFT)
        ttk.Button(buttons, text="播放输出", command=self.play_output).pack(side=LEFT, padx=8)
        ttk.Button(buttons, text="打开输出目录", command=self.open_output_folder).pack(side=LEFT)

    def _build_parameter_card(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="参数调节", style="Card.TLabel", font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(4, 4))
        self.speed_value_label = self._slider_row(
            parent, "🚀 语速 (Speed)", self.speed, -50, 100, "%", DEFAULT_SPEED
        )
        self.volume_value_label = self._slider_row(
            parent, "🔊 音量 (Volume)", self.volume, -50, 50, "%", DEFAULT_VOLUME
        )
        self.pitch_value_label = self._slider_row(
            parent, "🎵 音调 (Pitch)", self.pitch, -12, 12, "", DEFAULT_PITCH
        )

    def _slider_row(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.IntVar,
        from_: int,
        to: int,
        suffix: str,
        default: int,
    ) -> ttk.Label:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=5)
        header = ttk.Frame(row, style="Card.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text=label, style="Card.TLabel").pack(side=LEFT)
        value_label = ttk.Label(header, text=self._format_slider_value(variable.get(), suffix), style="Hint.TLabel")
        value_label.pack(side=RIGHT)
        scale = ttk.Scale(row, from_=from_, to=to, orient="horizontal", command=lambda value: self._on_slider_change(variable, value, value_label, suffix))
        scale.set(variable.get())
        scale.pack(fill="x", pady=(4, 2))
        footer = ttk.Frame(row, style="Card.TFrame")
        footer.pack(fill="x")
        ttk.Label(footer, text=str(from_), style="Hint.TLabel").pack(side=LEFT)
        ttk.Button(footer, text="重置", command=lambda: self._reset_slider(variable, scale, value_label, suffix, default)).pack(side=LEFT, padx=(150, 0))
        ttk.Label(footer, text=str(to), style="Hint.TLabel").pack(side=RIGHT)
        return value_label

    def _format_slider_value(self, value: int, suffix: str) -> str:
        sign = "+" if value > 0 else ""
        return f"{sign}{value}{suffix}"

    def _on_slider_change(self, variable: tk.IntVar, value: str, label: ttk.Label, suffix: str) -> None:
        int_value = int(round(float(value)))
        variable.set(int_value)
        label.configure(text=self._format_slider_value(int_value, suffix))

    def _reset_slider(self, variable: tk.IntVar, scale: ttk.Scale, label: ttk.Label, suffix: str, default: int) -> None:
        variable.set(default)
        scale.set(default)
        label.configure(text=self._format_slider_value(default, suffix))

    def _apply_voice_role(self, _event=None) -> None:
        audio, text = VOICE_ROLES.get(self.voice_role.get(), ("", ""))
        if not audio or not text:
            return
        self.ref_audio.set(str(APP_DIR / audio))
        self.ref_text.set(str(APP_DIR / text))

    def _build_install_card(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="环境管理", style="Card.TLabel", font=("Segoe UI", 15, "bold")).pack(anchor="w")
        ttk.Label(
            parent,
            text=(
                "推荐安装会安装官方 neutts 轮子（含 espeak-ng），避免本地 CMake/nmake 构建失败；"
                "GGUF/流式模型需要另装 llama-cpp-python。"
            ),
            style="Hint.TLabel",
            wraplength=360,
        ).pack(anchor="w", pady=(2, 12))
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x")
        self.install_button = ttk.Button(row, text="一键安装依赖", style="Accent.TButton", command=self.install_dependencies)
        self.install_button.pack(side=LEFT)
        self.gguf_button = ttk.Button(row, text="安装 GGUF 可选依赖", command=self.install_gguf_dependencies)
        self.gguf_button.pack(side=LEFT, padx=8)
        self.uninstall_button = ttk.Button(row, text="一键卸载", command=self.uninstall_dependencies)
        self.uninstall_button.pack(side=LEFT)
        self.progress = ttk.Progressbar(parent, mode="indeterminate")
        self.progress.pack(fill="x", pady=14)

    def _build_log_card(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="运行日志", style="Card.TLabel", font=("Segoe UI", 15, "bold")).pack(anchor="w")
        self.log = tk.Text(parent, height=24, wrap="word", bg="#020617", fg="#d1d5db", relief="flat", padx=10, pady=10, font=("Consolas", 9))
        self.log.pack(fill=BOTH, expand=True, pady=(8, 0))
        self._log("准备就绪。建议先点击“一键安装依赖”。")

    def _path_row(self, parent: ttk.Frame, label: str, variable: tk.StringVar, filetypes: list[tuple[str, str]], save: bool = False) -> None:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=5)
        ttk.Label(row, text=label, style="Card.TLabel", width=14).pack(side=LEFT)
        ttk.Entry(row, textvariable=variable).pack(side=LEFT, fill="x", expand=True, padx=8)
        command = lambda: self._choose_path(variable, filetypes, save)
        ttk.Button(row, text="浏览", command=command).pack(side=LEFT)

    def _choose_path(self, variable: tk.StringVar, filetypes: list[tuple[str, str]], save: bool) -> None:
        path = filedialog.asksaveasfilename(filetypes=filetypes, defaultextension=".wav") if save else filedialog.askopenfilename(filetypes=filetypes)
        if path:
            variable.set(path)

    def install_dependencies(self) -> None:
        # Install the published wheel instead of `pip install -e .`. The wheel
        # bundles espeak-ng, while editable installation invokes scikit-build /
        # CMake and fails on many Windows machines without nmake or C++ tools.
        # --only-binary prevents pip from falling back to a local source build.
        self._run_command([sys.executable, "-m", "pip", "install", "--only-binary=:all:", *CORE_DEPENDENCIES], "正在安装 NeuTTS 官方运行包...")

    def install_gguf_dependencies(self) -> None:
        if sys.platform.startswith("win"):
            self._log(
                "提示：Windows 安装 llama-cpp-python 如遇 Long Path 错误，"
                "请先启用系统长路径支持，或改用默认非 GGUF 模型。"
            )
        self._run_command([sys.executable, "-m", "pip", "install", "llama-cpp-python"], "正在安装 GGUF 可选依赖...")

    def uninstall_dependencies(self) -> None:
        if not messagebox.askyesno("确认卸载", "将卸载 NeuTTS 及常见可选依赖，是否继续？"):
            return
        self._run_command([sys.executable, "-m", "pip", "uninstall", "-y", *UNINSTALL_PACKAGES], "正在卸载依赖...")

    def generate_audio(self) -> None:
        # Read every Tk value on the UI thread before starting the worker.
        # Accessing Tk widgets/StringVar/IntVar from a background thread can
        # crash Tcl/Tk on Windows instead of raising a normal Python exception.
        config = {
            "text": self.text_input.get("1.0", END).strip(),
            "ref_audio": self.ref_audio.get(),
            "ref_text": self.ref_text.get().strip(),
            "output_path": self.output_path.get(),
            "backbone": self.backbone.get().strip(),
            "device": self.device.get().strip(),
            "speed": self.speed.get(),
            "volume": self.volume.get(),
            "pitch": self.pitch.get(),
        }
        if not config["text"]:
            messagebox.showwarning("缺少文本", "请输入要朗读的文本。")
            return
        self._run_python(lambda: self._generate_audio_task(config), "正在生成语音...")

    def _source_has_bundled_espeak(self) -> bool:
        source_pkg = APP_DIR / "neutts"
        if sys.platform.startswith("win"):
            return any(source_pkg.glob("espeak-ng*.dll"))
        if sys.platform == "darwin":
            return any(source_pkg.glob("libespeak-ng*.dylib"))
        return any(source_pkg.glob("libespeak-ng.so*")) or any(source_pkg.glob("libespeak-ng*.so"))

    def _import_neutts_class(self):
        # A downloaded source tree does not contain the bundled espeak-ng files
        # until it is built. Prefer the pip-installed NeuTTS wheel in that case,
        # because the wheel includes espeak-ng and avoids the "espeak not
        # installed on your system" runtime error on Windows.
        if not self._source_has_bundled_espeak():
            app_dir = str(APP_DIR)
            sys.path[:] = [entry for entry in sys.path if entry not in {"", app_dir}]
            for name in list(sys.modules):
                if name == "neutts" or name.startswith("neutts."):
                    del sys.modules[name]
            self._log_threadsafe("使用已安装的 NeuTTS 官方包（包含 espeak-ng）。")

        try:
            module = importlib.import_module("neutts")
        except RuntimeError as exc:
            if "espeak" in str(exc).lower():
                raise RuntimeError(
                    "未找到 espeak-ng。请先点击“一键安装依赖”安装官方 neutts 轮子；"
                    "不要使用 pip install -e .，因为本地源码未构建时不包含 espeak-ng。"
                ) from exc
            raise
        return module.NeuTTS

    def _generate_audio_task(self, config: dict[str, object]) -> None:
        import soundfile as sf
        import torch

        NeuTTS = self._import_neutts_class()

        text = str(config["text"])
        ref_audio = Path(str(config["ref_audio"])).expanduser()
        ref_text_value = str(config["ref_text"]).strip()
        output = Path(str(config["output_path"])).expanduser()
        backbone = str(config["backbone"])
        device = str(config["device"])
        speed = int(config["speed"])
        volume = int(config["volume"])
        pitch = int(config["pitch"])
        if not ref_audio.exists():
            raise FileNotFoundError(f"参考音频不存在: {ref_audio}")
        if Path(ref_text_value).expanduser().exists():
            ref_text_value = Path(ref_text_value).expanduser().read_text(encoding="utf-8").strip()
        if not ref_text_value:
            raise ValueError("参考文本不能为空。")
        output.parent.mkdir(parents=True, exist_ok=True)

        self._log_threadsafe("加载模型...")
        tts = NeuTTS(
            backbone_repo=backbone,
            backbone_device=device,
            codec_repo="neuphonic/neucodec",
            codec_device=device if device != "gpu" else "cpu",
        )
        cache_path = ref_audio.with_suffix(".pt")
        if cache_path.exists():
            self._log_threadsafe(f"加载已编码参考音频: {cache_path}")
            ref_codes = torch.load(cache_path, map_location="cpu")
        else:
            self._log_threadsafe("编码参考音频...")
            ref_codes = tts.encode_reference(str(ref_audio))
            torch.save(ref_codes, cache_path)
        self._log_threadsafe("生成音频...")
        wav = tts.infer(text, ref_codes, ref_text_value)
        wav = self._apply_audio_parameters(wav, tts.sample_rate, speed, pitch, volume)
        sf.write(output, wav, tts.sample_rate)
        self._log_threadsafe(f"完成: {output}")

    def _apply_audio_parameters(self, wav, sample_rate: int, speed: int, pitch: int, volume: int):
        import librosa
        import numpy as np

        audio = np.asarray(wav, dtype=np.float32)

        if speed:
            rate = max(0.25, 1.0 + speed / 100.0)
            self._log_threadsafe(f"应用语速: {self._format_slider_value(speed, '%')}")
            audio = librosa.effects.time_stretch(audio, rate=rate)
        if pitch:
            self._log_threadsafe(f"应用音调: {self._format_slider_value(pitch, '')} 半音")
            audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch)
        if volume:
            gain = 1.0 + volume / 100.0
            self._log_threadsafe(f"应用音量: {self._format_slider_value(volume, '%')}")
            audio = audio * gain

        return np.clip(audio, -1.0, 1.0)

    def _run_command(self, cmd: list[str], title: str) -> None:
        def task() -> None:
            process = subprocess.Popen(cmd, cwd=APP_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                self._log_threadsafe(line.rstrip())
            if process.wait() != 0:
                raise RuntimeError(f"命令失败: {' '.join(cmd)}")
        self._run_python(task, title)

    def _run_python(self, func, title: str) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("任务运行中", "请等待当前任务完成。")
            return
        self._set_busy(True)
        self._log(title)
        def wrapped() -> None:
            try:
                func()
                self.log_queue.put(("done", "任务完成。"))
            except Exception:
                self.log_queue.put(("error", traceback.format_exc()))
        self.worker = threading.Thread(target=wrapped, daemon=True)
        self.worker.start()

    def _set_busy(self, busy: bool) -> None:
        state = DISABLED if busy else NORMAL
        for button in (self.generate_button, self.install_button, self.gguf_button, self.uninstall_button):
            button.configure(state=state)
        self.progress.start(12) if busy else self.progress.stop()

    def _log(self, message: str) -> None:
        self.log.insert(END, message + "\n")
        self.log.see(END)

    def _log_threadsafe(self, message: str) -> None:
        self.log_queue.put(("log", message))

    def _drain_log_queue(self) -> None:
        try:
            while True:
                kind, message = self.log_queue.get_nowait()
                self._log(message)
                if kind in {"done", "error"}:
                    self._set_busy(False)
                    if kind == "error":
                        messagebox.showerror("任务失败", "请查看运行日志中的错误详情。")
        except queue.Empty:
            pass
        self.after(100, self._drain_log_queue)

    def play_output(self) -> None:
        path = Path(self.output_path.get()).expanduser()
        if not path.exists():
            messagebox.showwarning("文件不存在", f"未找到输出文件: {path}")
            return
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])

    def open_output_folder(self) -> None:
        folder = Path(self.output_path.get()).expanduser().parent
        folder.mkdir(parents=True, exist_ok=True)
        if sys.platform.startswith("win"):
            os.startfile(folder)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])


def main() -> None:
    NeuTTSGui().mainloop()


if __name__ == "__main__":
    main()
