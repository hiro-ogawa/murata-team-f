import argparse
from typing import List, Dict, Any, Optional

import sounddevice as sd
import numpy as np
from threading import Lock

# DEFAULT_DEVICE_NAME = "Miraisens Haptics Device: USB Audio (hw:1,0)"
DEFAULT_DEVICE_NAME = None

def list_sound_devices(as_json: bool = False) -> int:
    """List available audio devices using sounddevice.

    Returns 0 on success, non-zero on failure.
    """
    try:
        devices: List[Dict[str, Any]] = sd.query_devices()  # type: ignore[assignment]
        hostapis: List[Dict[str, Any]] = sd.query_hostapis()  # type: ignore[assignment]
    except Exception as e:
        print(f"デバイス情報の取得に失敗しました: {e}")
        return 1

    # 既定デバイス（インデックス）を取得（存在しない場合は None）
    default_in_idx = None
    default_out_idx = None
    try:
        d = sd.default.device
        if isinstance(d, (list, tuple)) and len(d) >= 2:
            default_in_idx, default_out_idx = d[0], d[1]
    except Exception:
        pass

    if as_json:
        # JSON 出力用に最小限のフィールドを整形
        out = []
        for idx, dev in enumerate(devices):
            hostapi_idx = dev.get("hostapi")
            hostapi_name = None
            if isinstance(hostapi_idx, int) and 0 <= hostapi_idx < len(hostapis):
                hostapi_name = hostapis[hostapi_idx].get("name")
            out.append(
                {
                    "index": idx,
                    "name": dev.get("name"),
                    "hostapi": hostapi_name or hostapi_idx,
                    "max_input_channels": dev.get("max_input_channels"),
                    "max_output_channels": dev.get("max_output_channels"),
                    "default_samplerate": dev.get("default_samplerate"),
                    "is_default_input": idx == default_in_idx,
                    "is_default_output": idx == default_out_idx,
                }
            )
        try:
            import json

            print(json.dumps(out, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"JSONの出力に失敗しました: {e}")
            return 1
        return 0

    # 表形式での出力
    header = f"{'Idx':>3}  {'HostAPI':<12} {'In':>3} {'Out':>3} {'Rate':>8}  Name"
    print(header)
    print("-" * len(header))

    for idx, dev in enumerate(devices):
        hostapi_idx = dev.get("hostapi")
        hostapi_name = str(hostapi_idx)
        if isinstance(hostapi_idx, int) and 0 <= hostapi_idx < len(hostapis):
            hostapi_name = hostapis[hostapi_idx].get("name", str(hostapi_idx))

        in_ch = dev.get("max_input_channels", 0)
        out_ch = dev.get("max_output_channels", 0)
        rate = dev.get("default_samplerate", "-")

        # 既定デバイスのマーキング
        markers = []
        if idx == default_in_idx:
            markers.append("IN")
        if idx == default_out_idx:
            markers.append("OUT")
        mark = (" [" + ",".join(markers) + "]") if markers else ""

        print(
            f"{idx:>3}  {hostapi_name:<12} {in_ch:>3} {out_ch:>3} {str(rate):>8}  {dev.get('name','')}" + mark
        )

    # 既定デバイス概要
    try:
        default_in_name = sd.query_devices(kind="input").get("name")
    except Exception:
        default_in_name = None
    try:
        default_out_name = sd.query_devices(kind="output").get("name")
    except Exception:
        default_out_name = None

    print()
    print(f"デフォルト入力: index={default_in_idx} name={default_in_name}")
    print(f"デフォルト出力: index={default_out_idx} name={default_out_name}")

    return 0


def _find_output_device_index_by_name(name_substring: str) -> Optional[int]:
    """部分一致で出力デバイスを検索し、インデックスを返す。見つからなければ None。"""
    try:
        devices: List[Dict[str, Any]] = sd.query_devices()  # type: ignore[assignment]
    except Exception:
        return None
    lower = name_substring.lower()
    for idx, d in enumerate(devices):
        if (d.get("max_output_channels", 0) or 0) > 0:
            if str(d.get("name", "")).lower().find(lower) >= 0:
                return idx
    return None


def play_dual_tone(
    left1_freq: float = 440.0,
    right1_freq: float = 441.0,
    left2_freq: float = 442.0,
    right2_freq: float = 443.0,
    samplerate: int = 44100,
    amplitude1: float = 0.2,
    amplitude2: float = 0.2,
    device_name: Optional[str] = DEFAULT_DEVICE_NAME,
    duration: float = 0.0,
) -> int:
    """左=left_freq, 右=right_freq のサイン波をリアルタイム生成して出力する。

    duration<=0 の場合は Ctrl+C まで無限再生。
    device_name が見つからなければデフォルト出力を使用。
    """
    # デバイス解決
    device_index: Optional[int] = None
    if device_name:
        device_index = _find_output_device_index_by_name(device_name)
        if device_index is None:
            print(f"指定デバイスが見つかりませんでした: '{device_name}'. デフォルト出力を使用します。")
    try:
        # 位相管理（コールバック内で更新）
        left1_phase = 0.0
        right1_phase = 0.0
        left2_phase = 0.0
        right2_phase = 0.0
        two_pi = 2.0 * np.pi
        left1_inc = two_pi * left1_freq / float(samplerate)
        right1_inc = two_pi * right1_freq / float(samplerate)
        left2_inc = two_pi * left2_freq / float(samplerate)
        right2_inc = two_pi * right2_freq / float(samplerate)

        def callback(outdata, frames, time, status):  # type: ignore[no-redef]
            nonlocal left1_phase, right1_phase, left2_phase, right2_phase
            if status:
                print(status, flush=False)
            t = np.arange(frames, dtype=np.float32)
            l1 = np.sin(left1_phase + left1_inc * t).astype(np.float32, copy=False)
            r1 = np.sin(right1_phase + right1_inc * t).astype(np.float32, copy=False)
            l2 = np.sin(left2_phase + left2_inc * t).astype(np.float32, copy=False)
            r2 = np.sin(right2_phase + right2_inc * t).astype(np.float32, copy=False)
            out = np.column_stack((amplitude1 * l1, amplitude1 * r1, amplitude2 * l2, amplitude2 * r2)).astype(np.float32, copy=False)
            outdata[:] = out
            left1_phase = (left1_phase + left1_inc * frames) % two_pi
            right1_phase = (right1_phase + right1_inc * frames) % two_pi
            left2_phase = (left2_phase + left2_inc * frames) % two_pi
            right2_phase = (right2_phase + right2_inc * frames) % two_pi

        with sd.OutputStream(
            samplerate=samplerate,
            channels=4,
            dtype="float32",
            device=device_index if device_index is not None else None,
            callback=callback,
        ) as stream:
            used_dev = stream.device
            try:
                dev_info = sd.query_devices(used_dev)
                dev_name = dev_info.get("name")
            except Exception:
                dev_name = str(used_dev)
            print(f"再生開始: device='{dev_name}'  fs={samplerate}Hz  L1={left1_freq}Hz R1={right1_freq}Hz L2={left2_freq}Hz R2={right2_freq}Hz amp1={amplitude1} amp2={amplitude2}")

            if duration and duration > 0:
                sd.sleep(int(duration * 1000))
            else:
                # 無限再生（100msごとにスリープ）
                try:
                    while True:
                        sd.sleep(100)
                except KeyboardInterrupt:
                    print("\n停止します。")
    except Exception as e:
        print(f"再生に失敗しました: {e}")
        return 1
    return 0


def run_gui(
    left1_freq: float = 440.0,
    right1_freq: float = 441.0,
    left2_freq: float = 442.0,
    right2_freq: float = 443.0,
    samplerate: int = 44100,
    amplitude1: float = 0.2,
    amplitude2: float = 0.2,
    frames: int = 1024,
    device_name: Optional[str] = DEFAULT_DEVICE_NAME,
) -> int:
    """FreeSimpleGUIでリサージュ図形を描画する簡易GUI。

    再生/停止ボタンを備え、起動時は停止状態。再生中は同一バッファを音声に出力。
    """
    try:
        import FreeSimpleGUI as sg  # type: ignore
    except Exception as e:
        print(f"FreeSimpleGUIのインポートに失敗しました: {e}\n'pip install freesimplegui' を実行してください。")
        return 1

    # 出力デバイス解決
    device_index: Optional[int] = None
    if device_name:
        device_index = _find_output_device_index_by_name(device_name)
        if device_index is None:
            print(f"指定デバイスが見つかりませんでした: '{device_name}'. デフォルト出力を使用します。")

    sg.theme("DarkBlue3")

    size_px = 400
    graph1 = sg.Graph(
        canvas_size=(size_px, size_px),
        graph_bottom_left=(-1.1, -1.1),
        graph_top_right=(1.1, 1.1),
        background_color="black",
        key="-GRAPH1-",
        enable_events=False,
    )
    graph2 = sg.Graph(
        canvas_size=(size_px, size_px),
        graph_bottom_left=(-1.1, -1.1),
        graph_top_right=(1.1, 1.1),
        background_color="black",
        key="-GRAPH2-",
        enable_events=False,
    )

    left_controls = [
        [sg.Text("Lissajous 1 (L1/R1)")],
        [sg.Text("L1 Hz"), sg.Input(str(left1_freq), size=(8, 1), key="-L1-FREQ-"), sg.Slider(range=(0.1, 880.0), default_value=left1_freq, resolution=0.1, orientation="h", size=(20, 15), enable_events=True, key="-L1-SLIDER-")],
        [sg.Text("R1 Hz"), sg.Input(str(right1_freq), size=(8, 1), key="-R1-FREQ-"), sg.Slider(range=(0.1, 880.0), default_value=right1_freq, resolution=0.1, orientation="h", size=(20, 15), enable_events=True, key="-R1-SLIDER-")],
        [sg.Text("Amp1"), sg.Input(f"{amplitude1:.2f}", size=(6, 1), key="-AMP1-"), sg.Slider(range=(0.0, 1.0), default_value=amplitude1, resolution=0.01, orientation="h", size=(20, 15), enable_events=True, key="-AMP1-SLIDER-")],
        [sg.Text("ϕL1"), sg.Input("0", size=(6, 1), key="-L1-PHASE-"), sg.Slider(range=(-180, 180), default_value=0, resolution=1, orientation="h", size=(20, 15), enable_events=True, key="-L1-PHASE-SLIDER-")],
        [sg.Text("ϕR1"), sg.Input("0", size=(6, 1), key="-R1-PHASE-"), sg.Slider(range=(-180, 180), default_value=0, resolution=1, orientation="h", size=(20, 15), enable_events=True, key="-R1-PHASE-SLIDER-")],
        [sg.Button("Apply L1R1", key="-APPLY-L1R1-")],
        [graph1],
    ]
    right_controls = [
        [sg.Text("Lissajous 2 (L2/R2)")],
        [sg.Text("L2 Hz"), sg.Input(str(left2_freq), size=(8, 1), key="-L2-FREQ-"), sg.Slider(range=(0.1, 880.0), default_value=left2_freq, resolution=0.1, orientation="h", size=(20, 15), enable_events=True, key="-L2-SLIDER-")],
        [sg.Text("R2 Hz"), sg.Input(str(right2_freq), size=(8, 1), key="-R2-FREQ-"), sg.Slider(range=(0.1, 880.0), default_value=right2_freq, resolution=0.1, orientation="h", size=(20, 15), enable_events=True, key="-R2-SLIDER-")],
        [sg.Text("Amp2"), sg.Input(f"{amplitude2:.2f}", size=(6, 1), key="-AMP2-"), sg.Slider(range=(0.0, 1.0), default_value=amplitude2, resolution=0.01, orientation="h", size=(20, 15), enable_events=True, key="-AMP2-SLIDER-")],
        [sg.Text("ϕL2"), sg.Input("0", size=(6, 1), key="-L2-PHASE-"), sg.Slider(range=(-180, 180), default_value=0, resolution=1, orientation="h", size=(20, 15), enable_events=True, key="-L2-PHASE-SLIDER-")],
        [sg.Text("ϕR2"), sg.Input("0", size=(6, 1), key="-R2-PHASE-"), sg.Slider(range=(-180, 180), default_value=0, resolution=1, orientation="h", size=(20, 15), enable_events=True, key="-R2-PHASE-SLIDER-")],
        [sg.Button("Apply L2R2", key="-APPLY-L2R2-")],
        [graph2],
    ]
    layout = [
        [sg.Button("Play", key="-PLAY-", button_color=("white", "#007acc")), sg.Button("Stop", key="-STOP-", disabled=True), sg.Button("Exit")],
        [sg.Column(left_controls), sg.Column(right_controls)],
    ]

    window = sg.Window("Lissajous Viewer", layout, finalize=True)

    # 座標軸（固定描画）
    axis_color = "#333333"
    graph1.draw_line((-1.05, 0.0), (1.05, 0.0), color=axis_color, width=1)
    graph1.draw_line((0.0, -1.05), (0.0, 1.05), color=axis_color, width=1)
    graph1.draw_rectangle((-1.0, -1.0), (1.0, 1.0), line_color=axis_color)
    graph2.draw_line((-1.05, 0.0), (1.05, 0.0), color=axis_color, width=1)
    graph2.draw_line((0.0, -1.05), (0.0, 1.05), color=axis_color, width=1)
    graph2.draw_rectangle((-1.0, -1.0), (1.0, 1.0), line_color=axis_color)

    # 2セット分のパラメータ
    two_pi = 2.0 * np.pi
    left1_phase = 0.0
    right1_phase = 0.0
    left2_phase = 0.0
    right2_phase = 0.0
    left1_inc = two_pi * left1_freq / float(samplerate)
    right1_inc = two_pi * right1_freq / float(samplerate)
    left2_inc = two_pi * left2_freq / float(samplerate)
    right2_inc = two_pi * right2_freq / float(samplerate)
    left1_offset = 0.0
    right1_offset = 0.0
    left2_offset = 0.0
    right2_offset = 0.0

    # コールバックとGUI間で最新バッファを共有
    latest_lr = None  # type: Optional[np.ndarray]
    buf_lock = Lock()

    # 再生管理
    stream: Optional[sd.OutputStream] = None
    playing = False

    def set_freqs(l1_new=None, r1_new=None, l2_new=None, r2_new=None):
        nonlocal left1_freq, right1_freq, left2_freq, right2_freq, left1_inc, right1_inc, left2_inc, right2_inc
        updated = False
        try:
            if l1_new is not None:
                left1_freq = min(880.0, max(0.1, float(l1_new)))
                updated = True
            if r1_new is not None:
                right1_freq = min(880.0, max(0.1, float(r1_new)))
                updated = True
            if l2_new is not None:
                left2_freq = min(880.0, max(0.1, float(l2_new)))
                updated = True
            if r2_new is not None:
                right2_freq = min(880.0, max(0.1, float(r2_new)))
                updated = True
        except Exception:
            pass
        if updated:
            left1_inc = two_pi * left1_freq / float(samplerate)
            right1_inc = two_pi * right1_freq / float(samplerate)
            left2_inc = two_pi * left2_freq / float(samplerate)
            right2_inc = two_pi * right2_freq / float(samplerate)
            window["-L1-FREQ-"].update(f"{left1_freq:.3f}")
            window["-R1-FREQ-"].update(f"{right1_freq:.3f}")
            window["-L2-FREQ-"].update(f"{left2_freq:.3f}")
            window["-R2-FREQ-"].update(f"{right2_freq:.3f}")
            print(f"周波数を更新: L1={left1_freq}Hz, R1={right1_freq}Hz, L2={left2_freq}Hz, R2={right2_freq}Hz")

    def set_phase(l1deg=None, r1deg=None, l2deg=None, r2deg=None):
        nonlocal left1_offset, right1_offset, left2_offset, right2_offset
        updated = False
        try:
            if l1deg is not None:
                left1_offset = np.deg2rad(min(180.0, max(-180.0, float(l1deg))))
                updated = True
            if r1deg is not None:
                right1_offset = np.deg2rad(min(180.0, max(-180.0, float(r1deg))))
                updated = True
            if l2deg is not None:
                left2_offset = np.deg2rad(min(180.0, max(-180.0, float(l2deg))))
                updated = True
            if r2deg is not None:
                right2_offset = np.deg2rad(min(180.0, max(-180.0, float(r2deg))))
                updated = True
        except Exception:
            pass
        if updated:
            window["-L1-PHASE-"].update(f"{np.rad2deg(left1_offset):.0f}")
            window["-R1-PHASE-"].update(f"{np.rad2deg(right1_offset):.0f}")
            window["-L2-PHASE-"].update(f"{np.rad2deg(left2_offset):.0f}")
            window["-R2-PHASE-"].update(f"{np.rad2deg(right2_offset):.0f}")
            print(f"位相を更新: ϕL1={np.rad2deg(left1_offset):.0f}°, ϕR1={np.rad2deg(right1_offset):.0f}°, ϕL2={np.rad2deg(left2_offset):.0f}°, ϕR2={np.rad2deg(right2_offset):.0f}°")

    def set_amp(a1_new=None, a2_new=None):
        nonlocal amplitude1, amplitude2
        try:
            if a1_new is not None:
                amplitude1 = min(1.0, max(0.0, float(a1_new)))
                window["-AMP1-"].update(f"{amplitude1:.2f}")
                print(f"振幅1を更新: Amp1={amplitude1}")
            if a2_new is not None:
                amplitude2 = min(1.0, max(0.0, float(a2_new)))
                window["-AMP2-"].update(f"{amplitude2:.2f}")
                print(f"振幅2を更新: Amp2={amplitude2}")
        except Exception as e:
            print(f"振幅の更新に失敗: {e}")

    def apply_freqs():
        try:
            l1 = float(window["-L1-FREQ-"].get())
            r1 = float(window["-R1-FREQ-"].get())
            l2 = float(window["-L2-FREQ-"].get())
            r2 = float(window["-R2-FREQ-"].get())
            set_freqs(l1, r1, l2, r2)
        except Exception as e:
            print(f"周波数の更新に失敗: {e}")
        try:
            a1 = float(window["-AMP1-"].get())
            a2 = float(window["-AMP2-"].get())
            set_amp(a1, a2)
        except Exception as e:
            print(f"振幅の更新に失敗: {e}")
        try:
            l1deg = float(window["-L1-PHASE-"].get())
            r1deg = float(window["-R1-PHASE-"].get())
            l2deg = float(window["-L2-PHASE-"].get())
            r2deg = float(window["-R2-PHASE-"].get())
            set_phase(l1deg, r1deg, l2deg, r2deg)
        except Exception as e:
            print(f"位相の更新に失敗: {e}")

    def generate_lr(n: int) -> np.ndarray:
        nonlocal left1_phase, right1_phase, left2_phase, right2_phase
        t = np.arange(n, dtype=np.float32)
        l1 = np.sin(left1_phase + left1_inc * t + left1_offset).astype(np.float32, copy=False)
        r1 = np.sin(right1_phase + right1_inc * t + right1_offset).astype(np.float32, copy=False)
        l2 = np.sin(left2_phase + left2_inc * t + left2_offset).astype(np.float32, copy=False)
        r2 = np.sin(right2_phase + right2_inc * t + right2_offset).astype(np.float32, copy=False)
        out = np.column_stack((amplitude1 * l1, amplitude1 * r1, amplitude2 * l2, amplitude2 * r2)).astype(np.float32, copy=False)
        left1_phase = (left1_phase + left1_inc * n) % two_pi
        right1_phase = (right1_phase + right1_inc * n) % two_pi
        left2_phase = (left2_phase + left2_inc * n) % two_pi
        right2_phase = (right2_phase + right2_inc * n) % two_pi
        return out

    def callback(outdata, frames_cb, time, status):  # type: ignore[no-redef]
        nonlocal latest_lr
        if status:
            print(status, flush=False)
        out = generate_lr(int(frames_cb))
        outdata[:] = out
        with buf_lock:
            latest_lr = out.copy()

    def start_stream():
        nonlocal stream, playing
        if playing:
            return
        try:
            stream = sd.OutputStream(
                samplerate=samplerate,
                channels=4,
                dtype="float32",
                device=device_index if device_index is not None else None,
                blocksize=frames,
                callback=callback,
            )
            stream.start()
            try:
                dev_info = sd.query_devices(stream.device)
                dev_name = dev_info.get("name")
            except Exception:
                dev_name = str(stream.device)
            print(
                f"再生開始(GUI): device='{dev_name}'  fs={samplerate}Hz  L1={left1_freq}Hz R1={right1_freq}Hz L2={left2_freq}Hz R2={right2_freq}Hz amp1={amplitude1} amp2={amplitude2}"
            )
            playing = True
            window["-PLAY-"].update(disabled=True)
            window["-STOP-"].update(disabled=False)
        except Exception as e:
            print(f"再生開始に失敗しました: {e}")
            stream = None
            playing = False

    def stop_stream():
        nonlocal stream, playing
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
            stream = None
        playing = False
        window["-PLAY-"].update(disabled=False)
        window["-STOP-"].update(disabled=True)

    last_figs1 = []  # Lissajous1描画ID
    last_figs2 = []  # Lissajous2描画ID
    target_points = 700

    def clear_curves():
        for fig in last_figs1:
            try:
                graph1.delete_figure(fig)
            except Exception:
                pass
        last_figs1.clear()
        for fig in last_figs2:
            try:
                graph2.delete_figure(fig)
            except Exception:
                pass
        last_figs2.clear()

    try:
        while True:
            event, values = window.read(timeout=16)
            if event in (sg.WIN_CLOSED, "Exit"):
                break
            if event == "-PLAY-":
                start_stream()
                continue
            if event == "-STOP-":
                stop_stream()
                clear_curves()
                continue
            if event == "-APPLY-L1R1-":
                try:
                    l1 = float(values["-L1-FREQ-"])
                    r1 = float(values["-R1-FREQ-"])
                    a1 = float(values["-AMP1-"])
                    l1deg = float(values["-L1-PHASE-"])
                    r1deg = float(values["-R1-PHASE-"])
                    set_freqs(l1_new=l1, r1_new=r1)
                    set_amp(a1_new=a1)
                    set_phase(l1deg=l1deg, r1deg=r1deg)
                except Exception as e:
                    print(f"L1R1適用エラー: {e}")
                continue
            if event == "-APPLY-L2R2-":
                try:
                    l2 = float(values["-L2-FREQ-"])
                    r2 = float(values["-R2-FREQ-"])
                    a2 = float(values["-AMP2-"])
                    l2deg = float(values["-L2-PHASE-"])
                    r2deg = float(values["-R2-PHASE-"])
                    set_freqs(l2_new=l2, r2_new=r2)
                    set_amp(a2_new=a2)
                    set_phase(l2deg=l2deg, r2deg=r2deg)
                except Exception as e:
                    print(f"L2R2適用エラー: {e}")
                continue
            if event == "-L1-SLIDER-":
                set_freqs(l1_new=values.get("-L1-SLIDER-"))
                continue
            if event == "-R1-SLIDER-":
                set_freqs(r1_new=values.get("-R1-SLIDER-"))
                continue
            if event == "-AMP1-SLIDER-":
                set_amp(a1_new=values.get("-AMP1-SLIDER-"))
                continue
            if event == "-L1-PHASE-SLIDER-":
                set_phase(l1deg=values.get("-L1-PHASE-SLIDER-"))
                continue
            if event == "-R1-PHASE-SLIDER-":
                set_phase(r1deg=values.get("-R1-PHASE-SLIDER-"))
                continue
            if event == "-L2-SLIDER-":
                set_freqs(l2_new=values.get("-L2-SLIDER-"))
                continue
            if event == "-R2-SLIDER-":
                set_freqs(r2_new=values.get("-R2-SLIDER-"))
                continue
            if event == "-AMP2-SLIDER-":
                set_amp(a2_new=values.get("-AMP2-SLIDER-"))
                continue
            if event == "-L2-PHASE-SLIDER-":
                set_phase(l2deg=values.get("-L2-PHASE-SLIDER-"))
                continue
            if event == "-R2-PHASE-SLIDER-":
                set_phase(r2deg=values.get("-R2-PHASE-SLIDER-"))
                continue

            # 再生中のみ描画。停止中は描画更新しない。
            if not playing:
                continue

            # 最新のオーディオバッファを取得（サウンドデバイスへ送ったものと同一）
            with buf_lock:
                data = None if latest_lr is None else latest_lr.copy()
            if data is None:
                continue

            # 既存曲線をクリア
            clear_curves()

            # Lissajous1点群
            step = max(1, (len(data) - 1) // target_points)
            pts1 = [(float(data[i, 0]), float(data[i, 1])) for i in range(0, len(data), step)]
            if len(pts1) >= 2:
                for i in range(len(pts1) - 1):
                    last_figs1.append(graph1.draw_line(pts1[i], pts1[i + 1], color="lime", width=1))
            # Lissajous2点群
            pts2 = [(float(data[i, 2]), float(data[i, 3])) for i in range(0, len(data), step)]
            if len(pts2) >= 2:
                for i in range(len(pts2) - 1):
                    last_figs2.append(graph2.draw_line(pts2[i], pts2[i + 1], color="cyan", width=1))
    except Exception as e:
        print(f"GUI/描画でエラー: {e}")
    finally:
        stop_stream()
        window.close()

    return 0


def main():
    parser = argparse.ArgumentParser(description="List audio devices via sounddevice")
    parser.add_argument(
        "--list",
        action="store_true",
        help="デバイス一覧を表示（デフォルト動作）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="JSON形式で出力",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="左440Hz/右441Hzでリアルタイム再生",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=f"使用する出力デバイス名（部分一致）。未指定時は '{DEFAULT_DEVICE_NAME}' を優先、なければデフォルト",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="再生時間[秒]。0以下で無限",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="FreeSimpleGUIでリサージュ図形を表示",
    )
    args = parser.parse_args()

    if args.gui:
        pref_name = args.device if args.device else DEFAULT_DEVICE_NAME
        rc = run_gui(device_name=pref_name)
        raise SystemExit(rc)

    if args.play:
        pref_name = args.device if args.device else DEFAULT_DEVICE_NAME
        return_code = play_dual_tone(
            left_freq=440.0,
            right_freq=441.0,
            samplerate=44100,
            amplitude=0.2,
            device_name=pref_name,
            duration=args.duration,
        )
        raise SystemExit(return_code)

    # 明示的に --list を付けなくても一覧表示
    return_code = list_sound_devices(as_json=args.json)
    raise SystemExit(return_code)


if __name__ == "__main__":
    main()
