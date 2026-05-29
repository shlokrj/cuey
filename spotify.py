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
    _run("""
tell application "Spotify"
    if player state is paused or player state is stopped then playpause
end tell
""")


def pause():
    _run("""
tell application "Spotify"
    if player state is playing then playpause
end tell
""")


def next_track():
    _run('tell application "Spotify" to next track')


def previous_track():
    _run('tell application "Spotify" to previous track')
