#!/usr/bin/env python3
"""Select local data and open the annotation GUI on Windows, macOS, or Linux."""
from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from pathlib import Path

APP = Path(__file__).resolve().parent


def read_config(app_dir):
    try:
        value = json.loads((app_dir / "launcher_config.json").read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def data_folder(path):
    path = Path(path).expanduser().resolve()
    if path.name == "chair_export":
        path = path.parent
    if not (path / "chair_export").is_dir():
        raise ValueError("chair_export を含むフォルダ（C31）を選択してください: " + str(path))
    return path


def choose_folder():
    """Native folder picker where Tk is available; terminal fallback otherwise."""
    window = None
    try:
        import tkinter
        from tkinter import filedialog
        window = tkinter.Tk()
        window.withdraw()
        selected = filedialog.askdirectory(title="chair_export を含むデータフォルダ（C31）を選択", mustexist=True)
        if not selected:
            raise KeyboardInterrupt
        return selected
    except (ImportError, RuntimeError) as exc:
        print("Folder picker unavailable:", exc)
    except Exception as exc:
        # A missing graphical display is an expected fallback, not a data error.
        print("Folder picker unavailable:", exc)
    finally:
        if window is not None:
            window.destroy()
    return input("Data folder containing chair_export: ").strip().strip('"')


def resolve_root(explicit=None, choose=False, app_dir=APP):
    if explicit:
        return data_folder(explicit)
    if not choose:
        # A self-contained transfer always uses its own bundled data first.
        candidates = [app_dir / "C31"]
        config = read_config(app_dir)
        if config.get("root"):
            candidates.append(Path(config["root"]))
        if len(app_dir.parents) > 2:
            candidates.append(app_dir.parents[2])
        for candidate in candidates:
            if (candidate / "chair_export").is_dir():
                return data_folder(candidate)
    return data_folder(choose_folder())


def save_config(root, app_dir=APP):
    try:
        (app_dir / "launcher_config.json").write_text(
            json.dumps({"root": str(root)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        print("Folder preference could not be saved; data saving is separate:", exc)


def run(argv=None):
    if sys.version_info < (3, 10):
        print("Python 3.10 or newer is required. https://www.python.org/downloads/")
        return 1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--choose-folder", action="store_true", help="Choose a different data folder")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--port", type=int, default=0, help="0 selects an available local port")
    args = parser.parse_args(argv)
    # Imports follow the version check so launchers can report old interpreters.
    from server import Store, create_server
    root = resolve_root(args.root, args.choose_folder)
    output = args.output or root / "Output" / "chair_continuous_annotations_v1"
    store = Store(root, output)
    server = create_server(store, args.port)
    save_config(root)
    url = f"http://127.0.0.1:{server.server_port}/"
    print("\nChair Log Annotation\n" + url + "\nData: " + str(root) + "\nAnnotations: " + str(output), flush=True)
    print("Keep this terminal open while editing. Stop with Ctrl+C.", flush=True)
    if not args.no_browser:
        def open_browser():
            try:
                if not webbrowser.open(url):
                    print("Open this address in your browser:", url, flush=True)
            except Exception as exc:
                print("Open this address in your browser:", url, exc, flush=True)
        timer = threading.Timer(0.4, open_browser)
        timer.daemon = True
        timer.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except KeyboardInterrupt:
        print("Cancelled.")
    except (OSError, ValueError) as exc:
        print("Could not start:", exc, file=sys.stderr)
        raise SystemExit(1)
