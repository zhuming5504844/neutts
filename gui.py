"""A small desktop GUI for running NeuTTS locally.

The GUI intentionally uses only the Python standard-library (tkinter) so it can
open before project dependencies are installed. Use the "Install dependencies"
button once after downloading the repository, then generate speech from the same
window.
"""

from __future__ import annotations

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
        self.backbone = tk.StringVar(value=BACKBONES[0])
        self.device = tk.StringVar(value="cpu")

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

        buttons = ttk.Frame(parent, style="Card.TFrame")
        buttons.pack(fill="x")
        self.generate_button = ttk.Button(buttons, text="开始生成语音", style="Accent.TButton", command=self.generate_audio)
        self.generate_button.pack(side=LEFT)
        ttk.Button(buttons, text="播放输出", command=self.play_output).pack(side=LEFT, padx=8)
        ttk.Button(buttons, text="打开输出目录", command=self.open_output_folder).pack(side=LEFT)

    def _build_install_card(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="环境管理", style="Card.TLabel", font=("Segoe UI", 15, "bold")).pack(anchor="w")
        ttk.Label(parent, text="安装按钮会执行 pip install -e .[all]；卸载会移除本项目和常见可选依赖。", style="Hint.TLabel", wraplength=360).pack(anchor="w", pady=(2, 12))
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x")
        self.install_button = ttk.Button(row, text="一键安装依赖", style="Accent.TButton", command=self.install_dependencies)
        self.install_button.pack(side=LEFT)
        self.uninstall_button = ttk.Button(row, text="一键卸载", command=self.uninstall_dependencies)
        self.uninstall_button.pack(side=LEFT, padx=8)
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
        self._run_command([sys.executable, "-m", "pip", "install", "-e", ".[all]"], "正在安装依赖...")

    def uninstall_dependencies(self) -> None:
        if not messagebox.askyesno("确认卸载", "将卸载 NeuTTS 及常见可选依赖，是否继续？"):
            return
        self._run_command([sys.executable, "-m", "pip", "uninstall", "-y", *UNINSTALL_PACKAGES], "正在卸载依赖...")

    def generate_audio(self) -> None:
        text = self.text_input.get("1.0", END).strip()
        if not text:
            messagebox.showwarning("缺少文本", "请输入要朗读的文本。")
            return
        self._run_python(lambda: self._generate_audio_task(text), "正在生成语音...")

    def _generate_audio_task(self, text: str) -> None:
        import soundfile as sf
        import torch
        from neutts import NeuTTS

        ref_audio = Path(self.ref_audio.get()).expanduser()
        ref_text_value = self.ref_text.get().strip()
        output = Path(self.output_path.get()).expanduser()
        if not ref_audio.exists():
            raise FileNotFoundError(f"参考音频不存在: {ref_audio}")
        if Path(ref_text_value).expanduser().exists():
            ref_text_value = Path(ref_text_value).expanduser().read_text(encoding="utf-8").strip()
        if not ref_text_value:
            raise ValueError("参考文本不能为空。")
        output.parent.mkdir(parents=True, exist_ok=True)

        self._log_threadsafe("加载模型...")
        tts = NeuTTS(
            backbone_repo=self.backbone.get().strip(),
            backbone_device=self.device.get().strip(),
            codec_repo="neuphonic/neucodec",
            codec_device=self.device.get().strip() if self.device.get().strip() != "gpu" else "cpu",
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
        sf.write(output, wav, 24000)
        self._log_threadsafe(f"完成: {output}")

    def _run_command(self, cmd: list[str], title: str) -> None:
        def task() -> None:
            self._log_threadsafe(title)
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
        for button in (self.generate_button, self.install_button, self.uninstall_button):
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
