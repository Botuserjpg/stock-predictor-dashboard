"""WSGI entry point for the refactored Stock Predictor Pro package.

Development:  python wsgi.py            (Flask-SocketIO dev server)
Production:   gunicorn -c gunicorn.conf.py wsgi:app
"""
import os
import sys

from dotenv import load_dotenv


_VENV_PY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".venv310", "Scripts", "python.exe"
)
if os.path.isfile(_VENV_PY) and (sys.executable or "").lower() != _VENV_PY.lower():
    # Re-launch under the project virtualenv (which has TensorFlow/Keras/Prophet).
    import subprocess

    try:
        sys.exit(subprocess.call([_VENV_PY] + sys.argv))
    except KeyboardInterrupt:
        # Ctrl+C is delivered to both the launcher and child on Windows. The
        # child shuts down the web server; the launcher should exit quietly.
        sys.exit(0)


# Load runtime settings and detect a local port conflict before Eventlet
# monkey-patches threading. Exiting after Eventlet is initialized can produce
# a noisy "greenlet is being finalized" traceback during interpreter teardown.
load_dotenv()


def _exit_if_port_in_use() -> None:
    """Print a clean, actionable message if the development port is occupied."""
    try:
        port = int(os.getenv("PORT", "5000"))
    except (TypeError, ValueError):
        port = 5000

    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(1)
        if probe.connect_ex(("127.0.0.1", port)) != 0:
            return

    print(
        f"[!] Port {port} is already in use — another instance of this app is running.\n"
        "    Close that terminal/window first (Ctrl+C), then run: python wsgi.py"
    )
    raise SystemExit(1)


if __name__ == "__main__":
    _exit_if_port_in_use()


# Eventlet's monkey-patched threading can emit ``greenlet is being finalized``
# during Ctrl+C shutdown on Windows. Threading mode is stable for local
# development; retain Eventlet for non-Windows production deployments.
WSGI_ASYNC_MODE = "threading"
if os.name != "nt":
    try:
        import eventlet

        eventlet.monkey_patch()
        import eventlet.green.thread  # noqa: F401 (sanity check the patch)
        WSGI_ASYNC_MODE = "eventlet"
    except Exception:  # noqa: BLE001 - any failure must not block startup
        pass

os.environ["WSGI_ASYNC_MODE"] = WSGI_ASYNC_MODE

from stockpredictor import create_app  # noqa: E402
from stockpredictor.extensions import socketio  # noqa: E402

app = create_app()

if __name__ == "__main__":
    host = app.config.get("HOST", "127.0.0.1")
    port = app.config.get("PORT", 5000)
    socketio.run(
        app,
        host=host,
        port=port,
        debug=app.config.get("DEBUG", False),
        allow_unsafe_werkzeug=True,
    )
