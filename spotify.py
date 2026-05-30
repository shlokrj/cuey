import subprocess


def _run(command: str):
    result = subprocess.run(
        ["osascript", "-e", command],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"[spotify] osascript error: {result.stderr.strip()}")


def play():
    _run('tell application "Spotify" to play')


def pause():
    _run('tell application "Spotify" to pause')


def next_track():
    _run('tell application "Spotify" to next track')


def previous_track():
    _run('tell application "Spotify" to previous track')
