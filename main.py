#!/usr/bin/env python3
"""
🔔 School Bell Cutter – GUI версия
Изисква: pip install customtkinter librosa pydub numpy scipy pygame
"""

import os
import sys
import threading
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox

# ──────────────────────────────────────────────
#  Проверка на зависимости
# ──────────────────────────────────────────────
missing = []
try:
    import librosa
except ImportError:
    missing.append("librosa")
try:
    from pydub import AudioSegment
    from pydub.effects import normalize
except ImportError:
    missing.append("pydub")
try:
    from scipy.signal import find_peaks
except ImportError:
    missing.append("scipy")

if missing:
    import tkinter as tk
    root = tk.Tk(); root.withdraw()
    messagebox.showerror("Липсващи пакети",
        f"Инсталирай с:\n\npip install {' '.join(missing)}")
    sys.exit(1)

try:
    import pygame
    # Инициализация с по-високо качество и по-голям буфер против накъсване
    pygame.mixer.pre_init(44100, -16, 2, 2048)
    pygame.mixer.init()
    PYGAME_OK = True
except Exception:
    PYGAME_OK = False

# ──────────────────────────────────────────────
#  Тема
# ──────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ACCENT   = "#3a86ff"
SUCCESS  = "#2ecc71"
WARNING  = "#f39c12"
DANGER   = "#e74c3c"
BG_CARD  = "#1e1e2e"
BG_MAIN  = "#13131f"
FG_MUTED = "#888899"

# ──────────────────────────────────────────────
#  АУДИО ЛОГИКА
# ──────────────────────────────────────────────

def fmt_time(s: float) -> str:
    m = int(s) // 60
    sec = s - m * 60
    return f"{m}:{sec:05.2f}"

def format_time_ms(s: float) -> str:
    """Форматира секунди в MM:SS.ss"""
    m = int(s) // 60
    sec = s - m * 60
    return f"{m:02d}:{sec:05.2f}"

def parse_time(ts: str) -> float:
    """Парсва MM:SS.ss или секунди обратно в float"""
    try:
        ts = ts.strip()
        if ":" in ts:
            parts = ts.split(":")
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3: # HH:MM:SS
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        return float(ts)
    except:
        return 0.0


def find_best_energy(y, sr, clip_dur, t_start, t_end):
    hop = 512
    rms   = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    mask  = (times >= t_start) & (times <= t_end - clip_dur)
    clip_f = int(clip_dur * sr / hop)
    scores = np.convolve(rms, np.ones(clip_f) / clip_f, mode='same')
    scores[~mask] = 0
    return times, scores


def find_best_beat(y, sr, clip_dur, t_start, t_end):
    hop   = 512
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times   = librosa.frames_to_time(beats, sr=sr)
    valid = beat_times[(beat_times >= t_start) & (beat_times <= t_end - clip_dur)]
    if len(valid) == 0:
        return find_best_energy(y, sr, clip_dur, t_start, t_end)
    rms   = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    mask  = (times >= t_start) & (times <= t_end - clip_dur)
    scores = np.zeros(len(times))
    for bt in valid:
        s = int(bt * sr); e = int((bt + clip_dur) * sr)
        if e > len(y): continue
        idx = np.argmin(np.abs(times - bt))
        scores[idx] = np.mean(np.abs(y[s:e]))
    scores[~mask] = 0
    return times, scores


def find_best_chorus(y, sr, clip_dur, t_start, t_end):
    hop    = 512
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    rec    = librosa.segment.recurrence_matrix(chroma, mode='affinity', sym=True)
    nov    = rec.sum(axis=1)
    times  = librosa.frames_to_time(np.arange(len(nov)), sr=sr, hop_length=hop)
    mask   = (times >= t_start) & (times <= t_end - clip_dur)
    clip_f = int(clip_dur * sr / hop)
    scores = np.convolve(nov, np.ones(clip_f) / clip_f, mode='same')
    scores[~mask] = 0
    return times, scores


def find_best_onset(y, sr, clip_dur, t_start, t_end):
    hop = 512
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop)
    mask = (times >= t_start) & (times <= t_end - clip_dur)
    clip_f = int(clip_dur * sr / hop)
    scores = np.convolve(onset_env, np.ones(clip_f) / clip_f, mode='same')
    scores[~mask] = 0
    return times, scores


def find_best_percussive(y, sr, clip_dur, t_start, t_end):
    hop = 512
    y_harm, y_perc = librosa.effects.hpss(y)
    rms = librosa.feature.rms(y=y_perc, frame_length=2048, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    mask = (times >= t_start) & (times <= t_end - clip_dur)
    clip_f = int(clip_dur * sr / hop)
    scores = np.convolve(rms, np.ones(clip_f) / clip_f, mode='same')
    scores[~mask] = 0
    return times, scores


def find_best_smart_bell(y, sr, clip_dur, t_start, t_end):
    hop = 512
    # 1. Силна атака (onset)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    # 2. Сила на звука (RMS)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    
    # Комбинираме с тежест: повече Onset, за да уловим "удара"
    scores_raw = onset_env * 0.7 + (rms / (np.max(rms) + 1e-9)) * 0.3
    
    times = librosa.frames_to_time(np.arange(len(scores_raw)), sr=sr, hop_length=hop)
    mask = (times >= t_start) & (times <= t_end - clip_dur)
    
    clip_f = int(clip_dur * sr / hop)
    # Конволюция за гладкост, но с тежест в началото (атаката)
    kernel = np.linspace(1.0, 0.2, clip_f) 
    scores = np.convolve(scores_raw, kernel, mode='same')
    
    scores[~mask] = 0
    return times, scores


def find_best_fusion(y, sr, clip_dur, t_start, t_end):
    # Вземаме сурови резултати
    t1, s1 = find_best_smart_bell(y, sr, clip_dur, t_start, t_end)
    _, s2 = find_best_percussive(y, sr, clip_dur, t_start, t_end)
    _, s3 = find_best_chorus(y, sr, clip_dur, t_start, t_end)
    _, s4 = find_best_onset(y, sr, clip_dur, t_start, t_end)
    
    # Нормализация
    s1 /= (np.max(s1) + 1e-9)
    s2 /= (np.max(s2) + 1e-9)
    s3 /= (np.max(s3) + 1e-9)
    s4 /= (np.max(s4) + 1e-9)
    
    # Гласуване (тегла)
    fusion = (s1 * 0.4) + (s4 * 0.3) + (s2 * 0.2) + (s3 * 0.1)
    return t1, fusion


METHODS = {
    "fusion":     ("🚀 Fusion – Комбинира Smart, Onset, Percussive и Chorus", find_best_fusion),
    "smart":      ("🔔 Smart Bell – Onset + RMS енергия", find_best_smart_bell),
    "beat":       ("🥁 Beat – Синхронизиран с темпо", find_best_beat),
    "energy":     ("⚡ Energy – Най-силна част", find_best_energy),
    "chorus":     ("🎵 Chorus – Структурен повтор", find_best_chorus),
    "onset":      ("🎯 Onset – Прецизно начало", find_best_onset),
    "percussive": ("🥁 Percussive – Само ударни", find_best_percussive),
}

def top_candidates(times, scores, clip_dur, sr, n=3):
    hop = 512
    gap = int(clip_dur * sr / hop)
    temp = scores.copy()
    results = []
    for _ in range(n):
        idx = np.argmax(temp)
        if temp[idx] <= 0: break
        results.append((times[idx], scores[idx]))
        lo = max(0, idx - gap // 2)
        hi = min(len(temp), idx + gap // 2)
        temp[lo:hi] = 0
    return results


def _find_and_set_ffmpeg():
    """
    Търси ffmpeg.exe и го задава по ВСИЧКИ начини, които pydub проверява.
    Реда на търсене:
      1. Папката на текущия .py файл (__file__)
      2. Папката на стартирания файл (sys.argv[0])
      3. System PATH
    """
    import shutil
    from pydub import AudioSegment
    from pydub import utils as pydub_utils

    ffmpeg_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"

    candidates = []
    try:
        candidates.append(os.path.dirname(os.path.abspath(__file__)))
    except Exception:
        pass
    try:
        candidates.append(os.path.dirname(os.path.abspath(sys.argv[0])))
    except Exception:
        pass

    for folder in candidates:
        path = os.path.join(folder, ffmpeg_name)
        if os.path.isfile(path):
            # Задай по ВСИЧКИ начини едновременно
            pydub_utils.converter   = path
            AudioSegment.converter  = path
            AudioSegment.ffmpeg     = path
            # ffprobe: ако има ffprobe.exe до ffmpeg, използвай него; иначе – ffmpeg
            ffprobe_path = os.path.join(folder, "ffprobe.exe" if sys.platform == "win32" else "ffprobe")
            if not os.path.isfile(ffprobe_path):
                ffprobe_path = path  # ffmpeg може да свърши работата
            pydub_utils.get_prober_name = lambda: ffprobe_path
            AudioSegment.ffprobe = ffprobe_path
            # Добави папката и в PATH на процеса
            os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
            return path

    # System PATH
    found = shutil.which("ffmpeg")
    if found:
        return found

    return None


# Задай ffmpeg веднага при зареждане на модула
_FFMPEG_PATH = _find_and_set_ffmpeg()


def check_ffmpeg():
    """Хвърля RuntimeError ако ffmpeg не е намерен."""
    if _FFMPEG_PATH is None:
        raise RuntimeError(
            "ffmpeg не е намерен!\n\n"
            "Постави ffmpeg.exe в същата папка като скрипта.\n"
            "Свали от: https://ffmpeg.org/download.html"
        )


def build_af_filter(fade_in, fade_out, dur_sec, opts: dict) -> str:
    """Строи ffmpeg -af филтър верига от аудио настройките."""
    filters = []

    # Сила (volume)
    vol_db = opts.get("volume_db", 0.0)
    if abs(vol_db) > 0.01:
        filters.append(f"volume={vol_db:.2f}dB")

    # Bass boost – low shelf @ 100 Hz
    bass_db = opts.get("bass_db", 0.0)
    if abs(bass_db) > 0.01:
        filters.append(f"equalizer=f=100:width_type=o:width=2:g={bass_db:.2f}")

    # Treble boost – high shelf @ 8000 Hz
    treble_db = opts.get("treble_db", 0.0)
    if abs(treble_db) > 0.01:
        filters.append(f"equalizer=f=8000:width_type=o:width=2:g={treble_db:.2f}")

    # Нормализация
    if opts.get("normalize", True):
        mode = opts.get("norm_mode", "lufs")
        if mode == "lufs":
            target = opts.get("norm_target", -14.0)
            # двупроходна loudnorm
            filters.append(f"loudnorm=I={target:.1f}:LRA=11:TP=-1.5")
        else:  # peak
            target = opts.get("norm_target", -1.0)
            filters.append(f"dynaudnorm=p=0.9:m=100")
            filters.append(f"volume={target:.2f}dB")

    # Fade in / out
    fade_out_start = max(0.0, dur_sec - fade_out)
    if fade_in > 0.01:
        filters.append(f"afade=t=in:d={fade_in:.3f}")
    if fade_out > 0.01:
        filters.append(f"afade=t=out:st={fade_out_start:.3f}:d={fade_out:.3f}")

    return ",".join(filters) if filters else "anull"


def cut_and_save(filepath, start_sec, dur_sec, output_path, fade_in, fade_out, audio_opts=None):
    """Реже аудио директно с ffmpeg – без pydub, без ffprobe."""
    import subprocess
    check_ffmpeg()
    if audio_opts is None:
        audio_opts = {}

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    af = build_af_filter(fade_in, fade_out, dur_sec, audio_opts)
    bitrate = audio_opts.get("bitrate", "192k")

    cmd = [
        _FFMPEG_PATH,
        "-y",
        "-ss", str(round(start_sec, 3)),
        "-i",  filepath,
        "-t",  str(round(dur_sec, 3)),
        "-af", af,
        "-ar", "44100",
        "-ac", "2",
    ]
    # Битрейт само за MP3/AAC
    ext = os.path.splitext(output_path)[1].lower()
    if ext in (".mp3", ".aac", ".m4a", ".ogg"):
        cmd += ["-b:a", bitrate]
    cmd.append(output_path)

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        err = result.stderr.decode(errors="replace")[-400:]
        raise RuntimeError(f"ffmpeg грешка:\n{err}")

    return dur_sec


# ──────────────────────────────────────────────
#  GUI
# ──────────────────────────────────────────────

class BellCutterApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🔔 School Bell Cutter")
        self.geometry("780x820")
        self.resizable(True, True)
        self.configure(fg_color=BG_MAIN)

        self._y = None
        self._sr = None
        self._total_dur = 0.0
        self._best_time = None
        self._candidates = []
        self._preview_file = None
        self._processing = False

        self._build_ui()
        self._check_ffmpeg_on_start()

    # ── UI builder ──────────────────────────────

    def _check_ffmpeg_on_start(self):
        """Показва статус за ffmpeg в заглавието при стартиране."""
        if _FFMPEG_PATH:
            self.title(f"🔔 School Bell Cutter  ✅ ffmpeg: {os.path.basename(_FFMPEG_PATH)}")
        else:
            self.title("🔔 School Bell Cutter  ❌ ffmpeg НЕ е намерен!")
            self.after(500, lambda: messagebox.showerror(
                "Липсва ffmpeg",
                "ffmpeg.exe не е намерен!\n\n"
                "Постави ffmpeg.exe в СЪЩАТА папка като main.py\n"
                f"(очаква се в: {os.path.dirname(os.path.abspath(sys.argv[0]))})"
            ))

    def _build_ui(self):
        # Заглавие
        header = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=0)
        header.pack(fill="x")
        ctk.CTkLabel(header, text="🔔  School Bell Cutter",
                     font=ctk.CTkFont(size=22, weight="bold"),
                     text_color=ACCENT).pack(pady=14)
        ctk.CTkLabel(header,
                     text="Автоматично намиране и рязане на най-добрия момент за звънец",
                     font=ctk.CTkFont(size=12), text_color=FG_MUTED).pack(pady=(0, 12))

        scroll = ctk.CTkScrollableFrame(self, fg_color=BG_MAIN)
        scroll.pack(fill="both", expand=True, padx=0, pady=0)

        # ── Секция: Входен файл ──
        self._section(scroll, "📂  Входен файл")
        row = ctk.CTkFrame(scroll, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=(0, 6))
        self.input_var = ctk.StringVar(value="Не е избран файл…")
        ctk.CTkEntry(row, textvariable=self.input_var,
                     state="readonly", width=520).pack(side="left", padx=(0, 8))
        ctk.CTkButton(row, text="Избери…", width=110,
                      command=self._pick_input).pack(side="left")

        # Информация за песента
        self.song_info = ctk.CTkLabel(scroll, text="", text_color=FG_MUTED,
                                      font=ctk.CTkFont(size=11))
        self.song_info.pack(padx=20, anchor="w")

        # ── Секция: Прозорец за търсене ──
        self._section(scroll, "🔍  Времеви прозорец за търсене")
        wf = ctk.CTkFrame(scroll, fg_color="transparent")
        wf.pack(fill="x", padx=20, pady=(0, 8))

        ctk.CTkLabel(wf, text="От (сек):", width=80).grid(row=0, column=0, sticky="w")
        self.start_var = ctk.StringVar(value="0")
        ctk.CTkEntry(wf, textvariable=self.start_var, width=90).grid(row=0, column=1, padx=(0,20))

        ctk.CTkLabel(wf, text="До (сек):", width=80).grid(row=0, column=2, sticky="w")
        self.end_var = ctk.StringVar(value="(края)")
        ctk.CTkEntry(wf, textvariable=self.end_var, width=90).grid(row=0, column=3, padx=(0,20))

        ctk.CTkLabel(wf, text="💡 Остави 'До' празно = цялата песен",
                     text_color=FG_MUTED, font=ctk.CTkFont(size=11)).grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(4,0))

        # ── Секция: Настройки ──
        self._section(scroll, "⚙️  Настройки")
        sf = ctk.CTkFrame(scroll, fg_color="transparent")
        sf.pack(fill="x", padx=20, pady=(0,8))

        # Дължина
        ctk.CTkLabel(sf, text="Дължина (сек):", width=150, anchor="w").grid(
            row=0, column=0, sticky="w", pady=4)
        self.dur_var = ctk.DoubleVar(value=30)
        dur_row = ctk.CTkFrame(sf, fg_color="transparent")
        dur_row.grid(row=0, column=1, columnspan=2, sticky="w")
        self.dur_slider = ctk.CTkSlider(dur_row, from_=5, to=120,
                                         variable=self.dur_var, width=280,
                                         command=self._update_dur_label)
        self.dur_slider.pack(side="left")
        self.dur_label = ctk.CTkLabel(dur_row, text="30.0 сек", width=70)
        self.dur_label.pack(side="left", padx=8)

        # Метод
        ctk.CTkLabel(sf, text="Метод:", width=150, anchor="w").grid(
            row=1, column=0, sticky="w", pady=4)
        self.method_var = ctk.StringVar(value="fusion")
        method_menu = ctk.CTkOptionMenu(
            sf, variable=self.method_var,
            values=list(METHODS.keys()),
            width=200)
        method_menu.grid(row=1, column=1, sticky="w")
        
        # Бутон за информация
        ctk.CTkButton(sf, text="ℹ️", width=40,
                      fg_color="#444466", hover_color="#555588",
                      command=self._show_algo_info).grid(row=1, column=1, sticky="e", padx=(0, 40))

        # Fade In
        ctk.CTkLabel(sf, text="Fade-in (сек):", width=150, anchor="w").grid(
            row=3, column=0, sticky="w", pady=4)
        self.fadein_var = ctk.DoubleVar(value=0.5)
        fi_row = ctk.CTkFrame(sf, fg_color="transparent")
        fi_row.grid(row=3, column=1, columnspan=2, sticky="w")
        ctk.CTkSlider(fi_row, from_=0, to=5, variable=self.fadein_var, width=280,
                       command=lambda v: self.fi_label.configure(
                           text=f"{float(v):.1f} сек")).pack(side="left")
        self.fi_label = ctk.CTkLabel(fi_row, text="0.5 сек", width=70)
        self.fi_label.pack(side="left", padx=8)

        # Fade Out
        ctk.CTkLabel(sf, text="Fade-out (сек):", width=150, anchor="w").grid(
            row=4, column=0, sticky="w", pady=4)
        self.fadeout_var = ctk.DoubleVar(value=1.5)
        fo_row = ctk.CTkFrame(sf, fg_color="transparent")
        fo_row.grid(row=4, column=1, columnspan=2, sticky="w")
        ctk.CTkSlider(fo_row, from_=0, to=5, variable=self.fadeout_var, width=280,
                       command=lambda v: self.fo_label.configure(
                           text=f"{float(v):.1f} сек")).pack(side="left")
        self.fo_label = ctk.CTkLabel(fo_row, text="1.5 сек", width=70)
        self.fo_label.pack(side="left", padx=8)

        # ── Секция: Аудио настройки ──
        self._section(scroll, "🎚️  Аудио настройки")
        af = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=10)
        af.pack(fill="x", padx=20, pady=(0, 8))
        af.columnconfigure(1, weight=1)

        def _slider_row(parent, row, label, var, from_, to_, default, fmt="{:.1f}", suffix="dB", width=260):
            ctk.CTkLabel(parent, text=label, width=155, anchor="w").grid(
                row=row, column=0, sticky="w", padx=12, pady=6)
            inner = ctk.CTkFrame(parent, fg_color="transparent")
            inner.grid(row=row, column=1, sticky="w", padx=(0,12))
            lbl = ctk.CTkLabel(inner, text=fmt.format(default) + f" {suffix}", width=80)
            
            # Динамично обновяване, което чете текущия suffix от sl
            def _upd(v, sl=None): 
                sl.unit_label.configure(text=f"{sl.fmt.format(float(v))} {sl.unit_suffix}")
            
            sl = ctk.CTkSlider(inner, from_=from_, to=to_, variable=var,
                               width=width)
            sl.configure(command=lambda v: _upd(v, sl))
            sl.pack(side="left")
            lbl.pack(side="left", padx=6)
            sl.unit_label = lbl 
            sl.unit_suffix = suffix
            sl.fmt = fmt
            return sl

        # Нормализация toggle + mode + target
        norm_row = ctk.CTkFrame(af, fg_color="transparent")
        norm_row.grid(row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(10,4))
        self.norm_var = ctk.BooleanVar(value=True)
        norm_cb = ctk.CTkCheckBox(norm_row, text="Нормализация", variable=self.norm_var,
                                   command=self._toggle_norm, width=140)
        norm_cb.pack(side="left")
        ctk.CTkLabel(norm_row, text="Режим:", width=60).pack(side="left", padx=(16,4))
        self.norm_mode_var = ctk.StringVar(value="lufs")
        norm_mode_menu = ctk.CTkOptionMenu(
            norm_row, variable=self.norm_mode_var,
            values=["lufs", "peak"], width=90,
            command=self._on_norm_mode)
        norm_mode_menu.pack(side="left")
        self.norm_mode_hint = ctk.CTkLabel(norm_row,
            text="  LUFS = loudness стандарт (препоръчително)",
            text_color=FG_MUTED, font=ctk.CTkFont(size=11))
        self.norm_mode_hint.pack(side="left")

        # Target slider  (LUFS: -6 до -23 | peak: -0.1 до -6)
        self.norm_target_var = ctk.DoubleVar(value=-14.0)
        self.norm_target_slider = _slider_row(
            af, 1, "Target ниво:", self.norm_target_var,
            -23, -6, -14.0, "{:.1f}", "LUFS")

        # Separator
        ctk.CTkFrame(af, fg_color="#333344", height=1).grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=12, pady=6)

        # Volume
        self.vol_var = ctk.DoubleVar(value=0.0)
        _slider_row(af, 3, "Сила (Volume):", self.vol_var, -12, 12, 0.0, "{:+.1f}", "dB")

        # Bass
        self.bass_var = ctk.DoubleVar(value=0.0)
        _slider_row(af, 4, "Баси (Bass):", self.bass_var, -10, 10, 0.0, "{:+.1f}", "dB")

        # Treble
        self.treble_var = ctk.DoubleVar(value=0.0)
        _slider_row(af, 5, "Високи (Treble):", self.treble_var, -10, 10, 0.0, "{:+.1f}", "dB")

        # Separator
        ctk.CTkFrame(af, fg_color="#333344", height=1).grid(
            row=6, column=0, columnspan=2, sticky="ew", padx=12, pady=6)

        # Битрейт
        bitrate_row = ctk.CTkFrame(af, fg_color="transparent")
        bitrate_row.grid(row=7, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 10))
        ctk.CTkLabel(bitrate_row, text="Битрейт (MP3):", width=155).pack(side="left")
        self.bitrate_var = ctk.StringVar(value="192k")
        for br in ["128k", "192k", "256k", "320k"]:
            ctk.CTkRadioButton(bitrate_row, text=br, value=br,
                               variable=self.bitrate_var, width=72).pack(side="left", padx=4)

        # ── Секция: Изходен файл ──
        self._section(scroll, "💾  Изходен файл")
        or_ = ctk.CTkFrame(scroll, fg_color="transparent")
        or_.pack(fill="x", padx=20, pady=(0, 8))
        self.output_var = ctk.StringVar(value="")
        ctk.CTkEntry(or_, textvariable=self.output_var,
                     placeholder_text="По подразбиране: <оригинал>_bell.mp3",
                     width=520).pack(side="left", padx=(0,8))
        ctk.CTkButton(or_, text="Запази като…", width=110,
                      command=self._pick_output).pack(side="left")

        # ── Бутон Анализирай ──
        self.analyze_btn = ctk.CTkButton(
            scroll, text="🔍  Анализирай песента", height=44,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=ACCENT, hover_color="#2563eb",
            command=self._start_analysis)
        self.analyze_btn.pack(padx=20, pady=12, fill="x")

        # ── Прогрес ──
        self.progress = ctk.CTkProgressBar(scroll, height=8)
        self.progress.pack(fill="x", padx=20, pady=(0, 4))
        self.progress.set(0)
        self.status_label = ctk.CTkLabel(scroll, text="", text_color=FG_MUTED,
                                          font=ctk.CTkFont(size=12))
        self.status_label.pack(padx=20, anchor="w")

        # ── Резултати ──
        self._section(scroll, "🏆  Топ кандидати")
        self.results_frame = ctk.CTkFrame(scroll, fg_color=BG_CARD, corner_radius=10)
        self.results_frame.pack(fill="x", padx=20, pady=(0, 8))
        self._placeholder = ctk.CTkLabel(self.results_frame,
            text="Тук ще се появят резултатите след анализа…",
            text_color=FG_MUTED, font=ctk.CTkFont(size=12))
        self._placeholder.pack(pady=18)

        # ── Бутон Изрежи ──
        self.cut_btn = ctk.CTkButton(
            scroll, text="✂️  Изрежи и запази", height=44,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#16a34a", hover_color="#15803d",
            state="disabled", command=self._do_cut)
        self.cut_btn.pack(padx=20, pady=(4, 16), fill="x")

    def _section(self, parent, title):
        ctk.CTkLabel(parent, text=title,
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color="#ccccdd").pack(padx=20, pady=(16, 6), anchor="w")

    # ── Callbacks ──────────────────────────────

    def _toggle_norm(self):
        state = "normal" if self.norm_var.get() else "disabled"
        self.norm_target_slider.configure(state=state)

    def _on_norm_mode(self, mode):
        # Update slider range and unit
        suffix = "LUFS" if mode == "lufs" else "dB"
        val = -14.0 if mode == "lufs" else -1.0
        
        self.norm_target_slider.configure(from_=-23 if mode=="lufs" else -6, 
                                          to=-6 if mode=="lufs" else -0.1)
        self.norm_target_var.set(val)
        
        # Update label text directly
        self.norm_target_slider.unit_label.configure(text=f"{val:.1f} {suffix}")
        
        hint = "  LUFS = loudness стандарт (препоръчително)" if mode == "lufs" else "  Peak = максимална стойност (dB)"
        self.norm_mode_hint.configure(text=hint)

    def _show_algo_info(self):
        popup = ctk.CTkToplevel(self)
        popup.title("Информация за алгоритмите")
        popup.geometry("600x450")
        popup.grab_set()
        
        info_text = (
            "🚀 Fusion: Комбинира най-доброто от всички методи чрез тегловно гласуване за максимална точност.\n\n"
            "🔔 Smart Bell: Съчетава ударни атаки (Onset detection) и обща енергия (RMS) за откриване на звуци с 'удар' като звънец.\n\n"
            "🥁 Beat: Анализира темпото на песента и търси най-подходящите удари.\n\n"
            "⚡ Energy: Фокусира се върху частите от песента с най-висока средна сила на звука.\n\n"
            "🎵 Chorus: Търси повтарящи се структури в песента (напр. припев).\n\n"
            "🎯 Onset: Идентифицира прецизното начало на звукови събития.\n\n"
            "🥁 Percussive: Изолира ударните елементи (барабани и удари) чрез спектрален анализ."
        )
        
        txt = ctk.CTkTextbox(popup, width=560, height=350, fg_color="transparent")
        txt.insert("0.0", info_text)
        txt.configure(state="disabled")
        txt.pack(padx=20, pady=20)
        
        ctk.CTkButton(popup, text="Затвори", command=popup.destroy).pack(pady=10)

    def _stop_preview(self):
        if PYGAME_OK:
            pygame.mixer.music.stop()

    def _update_dur_label(self, v):
        self.dur_label.configure(text=f"{float(v):.0f} сек")

    def _on_method_change(self, val):
        self.method_desc.configure(text=METHODS[val][0])

    def _pick_input(self):
        path = filedialog.askopenfilename(
            title="Избери аудио файл",
            filetypes=[("Аудио файлове", "*.mp3 *.wav *.flac *.ogg *.m4a *.aac"),
                       ("Всички файлове", "*.*")])
        if path:
            self.input_var.set(path)
            self.song_info.configure(text="⏳ Зареждам метаданни…")
            threading.Thread(target=self._load_meta, args=(path,), daemon=True).start()

    def _load_meta(self, path):
        try:
            y, sr = librosa.load(path, sr=None, mono=True)
            dur   = librosa.get_duration(y=y, sr=sr)
            self._y, self._sr, self._total_dur = y, sr, dur
            self.after(0, lambda: self.song_info.configure(
                text=f"⏱  Дължина: {fmt_time(dur)}  |  Честота: {sr} Hz  |  "
                     f"Файл: {os.path.basename(path)}"))
            self.after(0, lambda: self.end_var.set(f"{dur:.0f}"))
        except Exception as ex:
            self.after(0, lambda: self.song_info.configure(
                text=f"⚠️  Грешка: {ex}", text_color=DANGER))

    def _pick_output(self):
        path = filedialog.asksaveasfilename(
            title="Запази звънеца като…",
            defaultextension=".mp3",
            filetypes=[("MP3", "*.mp3"), ("WAV", "*.wav"),
                       ("OGG", "*.ogg"),  ("FLAC", "*.flac")])
        if path:
            self.output_var.set(path)

    # ── Анализ ─────────────────────────────────

    def _start_analysis(self):
        if not self.input_var.get() or self.input_var.get().startswith("Не е"):
            messagebox.showwarning("Внимание", "Избери входен аудио файл.")
            return
        if self._y is None:
            messagebox.showwarning("Внимание", "Изчакай зареждането на файла.")
            return
        if self._processing:
            return
        self._processing = True
        self.analyze_btn.configure(state="disabled", text="⏳ Анализирам…")
        self.cut_btn.configure(state="disabled")
        self._clear_results()
        self.progress.set(0)
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        try:
            self._status("Подготвям…", 0.05)
            clip_dur = float(self.dur_var.get())
            try:
                t_start = float(self.start_var.get())
            except Exception:
                t_start = 0.0
            try:
                t_end_raw = self.end_var.get().strip()
                t_end = float(t_end_raw) if t_end_raw not in ("", "(края)") else self._total_dur
            except Exception:
                t_end = self._total_dur

            t_end = min(t_end, self._total_dur)

            if t_end - t_start < clip_dur:
                raise ValueError(
                    f"Прозорецът ({t_end-t_start:.1f}s) е по-малък от исканата дължина ({clip_dur}s).")

            self._status("Анализирам аудио…", 0.3)
            method_fn = METHODS[self.method_var.get()][1]
            times, scores = method_fn(self._y, self._sr, clip_dur, t_start, t_end)

            self._status("Намирам топ кандидати…", 0.7)
            cands = top_candidates(times, scores, clip_dur, self._sr, n=5)

            self._candidates = cands
            self._best_time  = cands[0][0] if cands else 0.0

            self._status("Готово! ✅", 1.0)
            self.after(0, lambda: self._show_results(cands, clip_dur))
        except Exception as ex:
            msg = str(ex) if str(ex) and str(ex).strip() not in ("None", "") else repr(ex)
            self.after(0, lambda m=msg: messagebox.showerror("Грешка при анализ", m))
            self.after(0, lambda m=msg: self._status(f"❌ {m[:60]}", 0))
        finally:
            self._processing = False
            self.after(0, lambda: self.analyze_btn.configure(
                state="normal", text="🔍  Анализирай песента"))

    def _status(self, msg, pct):
        self.after(0, lambda: self.status_label.configure(text=msg))
        self.after(0, lambda: self.progress.set(pct))

    # ── Показване на резултатите ───────────────

    def _clear_results(self):
        for w in self.results_frame.winfo_children():
            w.destroy()

    def _show_results(self, cands, clip_dur):
        self._clear_results()
        if not cands:
            ctk.CTkLabel(self.results_frame, text="Не са намерени кандидати.",
                         text_color=DANGER).pack(pady=12)
            return

        self._selected_idx = 0 
        self._cand_vars = [] 
        self._cand_buttons = [] # Нулиране

        for rank, (t_orig, score) in enumerate(cands):
            # Променливи за този кандидат в MM:SS.ss формат
            sv = ctk.StringVar(value=format_time_ms(t_orig))
            ev = ctk.StringVar(value=format_time_ms(t_orig + clip_dur))
            self._cand_vars.append({
                "start": sv, "end": ev, "orig_start": t_orig, "orig_dur": clip_dur
            })

            outer = ctk.CTkFrame(self.results_frame, fg_color="#252535", corner_radius=10)
            outer.pack(fill="x", padx=10, pady=8)

            # Ред 1: Заглавие, Бар за резултат, Бутон за избор
            top_row = ctk.CTkFrame(outer, fg_color="transparent")
            top_row.pack(fill="x", padx=10, pady=(10, 5))

            badge_color = ACCENT if rank == 0 else "#444466"
            ctk.CTkLabel(top_row, text=f" #{rank + 1} ", fg_color=badge_color,
                         corner_radius=6, font=ctk.CTkFont(size=12, weight="bold"),
                         width=36).pack(side="left", padx=(0,10))

            # Score bar
            bar_frame = ctk.CTkFrame(top_row, fg_color="#1a1a2e", width=120, height=8, corner_radius=4)
            bar_frame.pack(side="left", padx=5)
            bar_frame.pack_propagate(False)
            norm = score / (cands[0][1] + 1e-9)
            ctk.CTkFrame(bar_frame, fg_color=ACCENT, width=int(120 * norm), height=8, corner_radius=4).place(x=0, y=0)

            # Select button
            sel_btn = ctk.CTkButton(
                top_row, text="✔ Избери", width=80, height=28,
                fg_color="#2d5016" if rank == 0 else "#333344",
                hover_color="#3d7020",
                command=lambda r=rank: self._select_candidate(r))
            sel_btn.pack(side="right", padx=5)
            self._cand_buttons.append(sel_btn)

            # Ред 2: Контроли за време
            ctrl_row = ctk.CTkFrame(outer, fg_color="#1e1e2e", corner_radius=6)
            ctrl_row.pack(fill="x", padx=10, pady=5)

            # Start controls
            ctk.CTkLabel(ctrl_row, text="Старт:", font=ctk.CTkFont(size=11)).pack(side="left", padx=(5,2))
            ctk.CTkButton(ctrl_row, text="-", width=24, height=24, command=lambda r=rank: self._adj_time(r, "start", -1)).pack(side="left", padx=1)
            ctk.CTkEntry(ctrl_row, textvariable=sv, width=70, height=24, font=ctk.CTkFont(size=12)).pack(side="left", padx=2)
            ctk.CTkButton(ctrl_row, text="+", width=24, height=24, command=lambda r=rank: self._adj_time(r, "start", 1)).pack(side="left", padx=1)

            # End controls
            ctk.CTkLabel(ctrl_row, text=" Край:", font=ctk.CTkFont(size=11)).pack(side="left", padx=(10,2))
            ctk.CTkButton(ctrl_row, text="-", width=24, height=24, command=lambda r=rank: self._adj_time(r, "end", -1)).pack(side="left", padx=1)
            ctk.CTkEntry(ctrl_row, textvariable=ev, width=70, height=24, font=ctk.CTkFont(size=12)).pack(side="left", padx=2)
            ctk.CTkButton(ctrl_row, text="+", width=24, height=24, command=lambda r=rank: self._adj_time(r, "end", 1)).pack(side="left", padx=1)

            # Reset button
            ctk.CTkButton(ctrl_row, text="🔄", width=32, height=24, fg_color="#444455", command=lambda r=rank: self._reset_cand(r)).pack(side="left", padx=(10,0))

            # Продължителност (динамично изчисляване)
            dv = ctk.StringVar()
            def _upd_dur(*args, s=sv, e=ev, d=dv):
                try:
                    dur = parse_time(e.get()) - parse_time(s.get())
                    if dur > 0:
                        d.set(f"⏱ {dur:.2f} сек")
                    else:
                        d.set("⚠️ Невалидно")
                except: d.set("⚠️ Грешка")
            
            sv.trace_add("write", _upd_dur)
            ev.trace_add("write", _upd_dur)
            _upd_dur() # Инициализация
            
            ctk.CTkLabel(ctrl_row, textvariable=dv, font=ctk.CTkFont(size=11, slant="italic"), 
                         text_color=SUCCESS).pack(side="left", padx=(15,0))

            # Ред 3: Play/Stop
            btn_row = ctk.CTkFrame(outer, fg_color="transparent")
            btn_row.pack(fill="x", padx=10, pady=(5, 10))

            if PYGAME_OK:
                ctk.CTkButton(
                    btn_row, text="▶ Пусни", width=74, height=28,
                    fg_color="#1a3a5c", hover_color="#1e4d80",
                    command=lambda r=rank: self._preview_cand(r)
                ).pack(side="left", padx=(0, 5))
                
                ctk.CTkButton(
                    btn_row, text="⏹ Спри", width=74, height=28,
                    fg_color="#5c1a1a", hover_color="#801e1e",
                    command=self._stop_preview
                ).pack(side="left")

        self._best_time = cands[0][0] # Първоначално задаване
        self.cut_btn.configure(state="normal")

    def _adj_time(self, rank, field, delta):
        try:
            var = self._cand_vars[rank][field]
            val = parse_time(var.get()) + delta
            var.set(format_time_ms(max(0, val)))
        except: pass

    def _reset_cand(self, rank):
        data = self._cand_vars[rank]
        data["start"].set(format_time_ms(data["orig_start"]))
        data["end"].set(format_time_ms(data["orig_start"] + data["orig_dur"]))

    def _select_candidate(self, rank):
        self._selected_idx = rank
        for i, btn in enumerate(self._cand_buttons):
            btn.configure(fg_color="#2d5016" if i == rank else "#333344",
                          text="✔ Избран" if i == rank else "✔ Избери")

    def _preview_cand(self, rank):
        try:
            start = parse_time(self._cand_vars[rank]["start"].get())
            end = parse_time(self._cand_vars[rank]["end"].get())
            dur = end - start
            if dur <= 0: return
            self._preview(start, dur)
        except: pass

    def _preview(self, start_sec, dur_sec):
        if not PYGAME_OK:
            return
        
        # Вземи текущите аудио настройки, за да се чуват и в прегледа
        audio_settings = {
            "volume_db":   self.vol_var.get(),
            "bass_db":     self.bass_var.get(),
            "treble_db":   self.treble_var.get(),
            "normalize":   self.norm_var.get(),
            "norm_mode":   self.norm_mode_var.get(),
            "norm_target": self.norm_target_var.get()
        }
        fade_in  = float(self.fadein_var.get())
        fade_out = float(self.fadeout_var.get())

        def worker():
            try:
                check_ffmpeg()
                script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
                cache_dir  = os.path.join(script_dir, "song_cache")
                os.makedirs(cache_dir, exist_ok=True)
                preview_path = os.path.join(cache_dir, "preview.wav")

                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
                import time; time.sleep(0.15)

                # Използваме същата логика за филтри като при запис
                af = build_af_filter(fade_in, fade_out, dur_sec, audio_settings)

                cmd = [
                    _FFMPEG_PATH, "-y", "-ss", str(round(start_sec, 3)),
                    "-i",  self.input_var.get(),
                    "-t",  str(round(dur_sec, 3)),
                    "-af", af,
                    "-ar", "44100", "-ac", "2", preview_path
                ]
                import subprocess
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode != 0:
                    raise RuntimeError(result.stderr.decode(errors="replace")[-300:])

                pygame.mixer.music.load(preview_path)
                pygame.mixer.music.play()
            except Exception as ex:
                msg = str(ex) if str(ex) and str(ex).strip() not in ("None", "") else repr(ex)
                self.after(0, lambda m=msg: messagebox.showwarning("Предварителен преглед", f"Грешка: {m}"))

        threading.Thread(target=worker, daemon=True).start()

    # ── Рязане ─────────────────────────────────

    def _do_cut(self):
        try:
            rank = self._selected_idx
            start = parse_time(self._cand_vars[rank]["start"].get())
            end = parse_time(self._cand_vars[rank]["end"].get())
            clip_dur = end - start
            if clip_dur <= 0:
                messagebox.showwarning("Внимание", "Невалидна дължина.")
                return
        except:
            messagebox.showwarning("Внимание", "Провери времената на избрания кандидат.")
            return

        filepath = self.input_var.get()
        output   = self.output_var.get().strip()
        if not output:
            base, _ = os.path.splitext(filepath)
            output  = f"{base}_bell.mp3"

        fade_in  = float(self.fadein_var.get())
        fade_out = float(self.fadeout_var.get())

        audio_settings = {
            "volume_db":   self.vol_var.get(),
            "bass_db":     self.bass_var.get(),
            "treble_db":   self.treble_var.get(),
            "normalize":   self.norm_var.get(),
            "norm_mode":   self.norm_mode_var.get(),
            "norm_target": self.norm_target_var.get(),
            "bitrate":     self.bitrate_var.get()
        }

        self.cut_btn.configure(state="disabled", text="⏳ Режа…")
        self._status("Режа и запазвам…", 0.5)

        def worker():
            try:
                real_len = cut_and_save(filepath, start, clip_dur, output, fade_in, fade_out, audio_settings)
                self.after(0, lambda: self._on_cut_done(output, start, real_len))
            except Exception as ex:
                err_msg = str(ex) if str(ex) and str(ex).strip() != "None" else repr(ex)
                self.after(0, lambda m=err_msg: messagebox.showerror("Грешка при рязане", m))
                self.after(0, lambda m=err_msg: self._status(f"❌ {m[:60]}", 0))
            finally:
                self.after(0, lambda: self.cut_btn.configure(
                    state="normal", text="✂️  Изрежи и запази"))

        threading.Thread(target=worker, daemon=True).start()

    def _on_cut_done(self, output, start, real_len):
        self.progress.set(1.0)
        self.status_label.configure(text=f"✅ Готово! Запазен: {os.path.basename(output)}")

        # Диалог за успех
        msg = (f"✅ Звънецът е запазен успешно!\n\n"
               f"📁 Файл:    {output}\n"
               f"⏱  Начало:  {fmt_time(start)}\n"
               f"⏱  Дължина: {real_len:.2f} сек\n"
               f"🔊 Fade-in: {self.fadein_var.get():.1f}s  |  "
               f"Fade-out: {self.fadeout_var.get():.1f}s")

        popup = ctk.CTkToplevel(self)
        popup.title("Успешно!")
        popup.geometry("420x220")
        popup.grab_set()
        ctk.CTkLabel(popup, text=msg, justify="left",
                     font=ctk.CTkFont(size=12)).pack(padx=20, pady=20)

        btn_row = ctk.CTkFrame(popup, fg_color="transparent")
        btn_row.pack(pady=4)

        if PYGAME_OK:
            def play_result():
                try:
                    pygame.mixer.music.load(output)
                    pygame.mixer.music.play()
                except Exception:
                    pass
            ctk.CTkButton(btn_row, text="▶ Пусни", width=100,
                           command=play_result).pack(side="left", padx=6)

        ctk.CTkButton(btn_row, text="OK", width=100,
                       fg_color=ACCENT, command=popup.destroy).pack(side="left", padx=6)


# ──────────────────────────────────────────────
#  СТАРТ
# ──────────────────────────────────────────────

if __name__ == "__main__":
    app = BellCutterApp()
    app.mainloop()