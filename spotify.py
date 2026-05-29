import subprocess


def _run(command: str):
    subprocess.run(["osascript", "-e", command], check=False)


def play():
    _run('tell application "Spotify" to play')


def pause():
    _run('tell application "Spotify" to pause')


def next_track():
    _run('tell application "Spotify" to next track')


def previous_track():
    _run('tell application "Spotify" to previous track')
