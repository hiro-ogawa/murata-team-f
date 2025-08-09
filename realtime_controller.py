import argparse
from typing import List, Dict, Any, Optional

import sounddevice as sd
import numpy as np
from threading import Lock


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
    left_freq: float = 440.0,
    right_freq: float = 441.0,
    samplerate: int = 44100,
    amplitude: float = 0.2,
    device_name: Optional[str] = "MacBook Pro Speakers",
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
        left_phase = 0.0
        right_phase = 0.0
        two_pi = 2.0 * np.pi
        left_inc = two_pi * left_freq / float(samplerate)
        right_inc = two_pi * right_freq / float(samplerate)

        def callback(outdata, frames, time, status):  # type: ignore[no-redef]
            nonlocal left_phase, right_phase
            if status:
                # 低頻度の XRuns などは通知のみ
                print(status, flush=False)
            t = np.arange(frames, dtype=np.float32)
            l = np.sin(left_phase + left_inc * t).astype(np.float32, copy=False)
            r = np.sin(right_phase + right_inc * t).astype(np.float32, copy=False)
            # 振幅を適用し、ステレオにパック
            out = amplitude * np.column_stack((l, r)).astype(np.float32, copy=False)
            outdata[:] = out
            # 位相を進める（オーバーフロー防止にモジュロ）
            left_phase = (left_phase + left_inc * frames) % two_pi
            right_phase = (right_phase + right_inc * frames) % two_pi

        with sd.OutputStream(
            samplerate=samplerate,
            channels=2,
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
            print(f"再生開始: device='{dev_name}'  fs={samplerate}Hz  L={left_freq}Hz  R={right_freq}Hz  amp={amplitude}")

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
    left_freq: float = 440.0,
    right_freq: float = 441.0,
    samplerate: int = 44100,
    amplitude: float = 0.2,
    frames: int = 1024,
    device_name: Optional[str] = "MacBook Pro Speakers",
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

    size_px = 500
    graph = sg.Graph(
        canvas_size=(size_px, size_px),
        graph_bottom_left=(-1.1, -1.1),
        graph_top_right=(1.1, 1.1),
        background_color="black",
        key="-GRAPH-",
        enable_events=False,
    )

    layout = [
        [
            sg.Text("Lissajous (L=440Hz, R=441Hz, fs=44.1kHz)"),
            sg.Push(),
            sg.Button("Play", key="-PLAY-", button_color=("white", "#007acc")),
            sg.Button("Stop", key="-STOP-", disabled=True),
            sg.Button("Exit"),
        ],
        [
            sg.Text("Left Hz"),
            sg.Input(str(left_freq), size=(8, 1), key="-L-FREQ-"),
            sg.Text("Right Hz"),
            sg.Input(str(right_freq), size=(8, 1), key="-R-FREQ-"),
            sg.Button("Apply", key="-APPLY-FREQ-"),
        ],
        [
            sg.Text("L"),
            sg.Slider(range=(0.1, 880.0), default_value=left_freq, resolution=0.1, orientation="h", size=(40, 15), enable_events=True, key="-L-SLIDER-")
        ],
        [
            sg.Text("R"),
            sg.Slider(range=(0.1, 880.0), default_value=right_freq, resolution=0.1, orientation="h", size=(40, 15), enable_events=True, key="-R-SLIDER-")
        ],
        [
            sg.Text("Amp"),
            sg.Input(f"{amplitude:.2f}", size=(6, 1), key="-AMP-"),
            sg.Slider(range=(0.0, 1.0), default_value=amplitude, resolution=0.01, orientation="h", size=(40, 15), enable_events=True, key="-AMP-SLIDER-"),
        ],
        [
            sg.Text("Phase L (deg)"),
            sg.Input("0", size=(6, 1), key="-L-PHASE-"),
            sg.Text("Phase R (deg)"),
            sg.Input("0", size=(6, 1), key="-R-PHASE-"),
        ],
        [
            sg.Text("ϕL"),
            sg.Slider(range=(-180, 180), default_value=0, resolution=1, orientation="h", size=(40, 15), enable_events=True, key="-L-PHASE-SLIDER-"),
        ],
        [
            sg.Text("ϕR"),
            sg.Slider(range=(-180, 180), default_value=0, resolution=1, orientation="h", size=(40, 15), enable_events=True, key="-R-PHASE-SLIDER-"),
        ],
        [graph],
    ]

    window = sg.Window("Lissajous Viewer", layout, finalize=True)

    # 座標軸（固定描画）
    axis_color = "#333333"
    graph.draw_line((-1.05, 0.0), (1.05, 0.0), color=axis_color, width=1)
    graph.draw_line((0.0, -1.05), (0.0, 1.05), color=axis_color, width=1)
    graph.draw_rectangle((-1.0, -1.0), (1.0, 1.0), line_color=axis_color)

    # 位相とインクリメント
    two_pi = 2.0 * np.pi
    left_phase = 0.0
    right_phase = 0.0
    left_inc = two_pi * left_freq / float(samplerate)
    right_inc = two_pi * right_freq / float(samplerate)

    # 位相オフセット（ラジアン、ユーザー操作で変更）
    left_offset = 0.0
    right_offset = 0.0

    # コールバックとGUI間で最新バッファを共有
    latest_lr = None  # type: Optional[np.ndarray]
    buf_lock = Lock()

    # 再生管理
    stream: Optional[sd.OutputStream] = None
    playing = False

    def set_freqs(lf_new: Optional[float] = None, rf_new: Optional[float] = None):
        nonlocal left_freq, right_freq, left_inc, right_inc
        updated = False
        try:
            if lf_new is not None:
                lf = float(lf_new)
                lf = min(880.0, max(0.1, lf))
                left_freq = lf
                updated = True
        except Exception:
            pass
        try:
            if rf_new is not None:
                rf = float(rf_new)
                rf = min(880.0, max(0.1, rf))
                right_freq = rf
                updated = True
        except Exception:
            pass
        if updated:
            left_inc = two_pi * left_freq / float(samplerate)
            right_inc = two_pi * right_freq / float(samplerate)
            # ウィジェット同期
            window["-L-FREQ-"].update(f"{left_freq:.3f}")
            window["-R-FREQ-"].update(f"{right_freq:.3f}")
            window["-L-SLIDER-"].update(left_freq)
            window["-R-SLIDER-"].update(right_freq)
            print(f"周波数を更新: L={left_freq}Hz, R={right_freq}Hz")

    def set_phase(ldeg_new: Optional[float] = None, rdeg_new: Optional[float] = None):
        nonlocal left_offset, right_offset
        updated = False
        try:
            if ldeg_new is not None:
                ld = float(ldeg_new)
                ld = min(180.0, max(-180.0, ld))
                left_offset = np.deg2rad(ld)
                updated = True
        except Exception:
            pass
        try:
            if rdeg_new is not None:
                rd = float(rdeg_new)
                rd = min(180.0, max(-180.0, rd))
                right_offset = np.deg2rad(rd)
                updated = True
        except Exception:
            pass
        if updated:
            # ウィジェット同期（テキストとスライダー）
            window["-L-PHASE-"].update(f"{np.rad2deg(left_offset):.0f}")
            window["-R-PHASE-"].update(f"{np.rad2deg(right_offset):.0f}")
            window["-L-PHASE-SLIDER-"].update(value=float(np.rad2deg(left_offset)))
            window["-R-PHASE-SLIDER-"].update(value=float(np.rad2deg(right_offset)))
            print(f"位相を更新: ϕL={np.rad2deg(left_offset):.0f}°, ϕR={np.rad2deg(right_offset):.0f}°")

    def set_amp(a_new: Optional[float] = None):
        nonlocal amplitude
        try:
            if a_new is None:
                return
            a = float(a_new)
            # 0.0〜1.0にクリップ
            a = min(1.0, max(0.0, a))
            amplitude = a
            # ウィジェット同期
            window["-AMP-"].update(f"{amplitude:.2f}")
            window["-AMP-SLIDER-"].update(amplitude)
            print(f"振幅を更新: Amp={amplitude}")
        except Exception as e:
            print(f"振幅の更新に失敗: {e}")

    def apply_freqs():
        # 既存: 周波数/振幅を適用
        try:
            lf = float(window["-L-FREQ-"].get())
            rf = float(window["-R-FREQ-"].get())
            set_freqs(lf, rf)
        except Exception as e:
            print(f"周波数の更新に失敗: {e}")
        try:
            a = float(window["-AMP-"].get())
            set_amp(a)
        except Exception as e:
            print(f"振幅の更新に失敗: {e}")
        # 新規: 位相適用
        try:
            ldeg = float(window["-L-PHASE-"].get())
            rdeg = float(window["-R-PHASE-"].get())
            set_phase(ldeg, rdeg)
        except Exception as e:
            print(f"位相の更新に失敗: {e}")

    def generate_lr(n: int) -> np.ndarray:
        nonlocal left_phase, right_phase
        t = np.arange(n, dtype=np.float32)
        l = np.sin(left_phase + left_inc * t + left_offset).astype(np.float32, copy=False)
        r = np.sin(right_phase + right_inc * t + right_offset).astype(np.float32, copy=False)
        out = amplitude * np.column_stack((l, r)).astype(np.float32, copy=False)
        left_phase = (left_phase + left_inc * n) % two_pi
        right_phase = (right_phase + right_inc * n) % two_pi
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
                channels=2,
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
                f"再生開始(GUI): device='{dev_name}'  fs={samplerate}Hz  L={left_freq}Hz  R={right_freq}Hz  amp={amplitude}"
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

    last_figs = []  # 描画した図形ID群（次フレームで削除）
    target_points = 700  # 描画負荷低減のための間引き目標数

    def clear_curves():
        for fig in last_figs:
            try:
                graph.delete_figure(fig)
            except Exception:
                pass
        last_figs.clear()

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
            if event == "-APPLY-FREQ-":
                apply_freqs()
                continue
            if event == "-L-SLIDER-":
                set_freqs(lf_new=values.get("-L-SLIDER-"))
                continue
            if event == "-R-SLIDER-":
                set_freqs(rf_new=values.get("-R-SLIDER-"))
                continue
            if event == "-AMP-SLIDER-":
                set_amp(values.get("-AMP-SLIDER-"))
                continue
            if event == "-L-PHASE-SLIDER-":
                set_phase(ldeg_new=values.get("-L-PHASE-SLIDER-"))
                continue
            if event == "-R-PHASE-SLIDER-":
                set_phase(rdeg_new=values.get("-R-PHASE-SLIDER-"))
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

            # 点群作成（間引き）
            step = max(1, (len(data) - 1) // target_points)
            pts = [(float(data[i, 0]), float(data[i, 1])) for i in range(0, len(data), step)]
            if len(pts) >= 2:
                # 折れ線で描画
                for i in range(len(pts) - 1):
                    last_figs.append(graph.draw_line(pts[i], pts[i + 1], color="lime", width=1))
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
        help="使用する出力デバイス名（部分一致）。未指定時は 'MacBook Pro Speakers' を優先、なければデフォルト",
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
        pref_name = args.device if args.device else "MacBook Pro Speakers"
        rc = run_gui(device_name=pref_name)
        raise SystemExit(rc)

    if args.play:
        pref_name = args.device if args.device else "MacBook Pro Speakers"
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
