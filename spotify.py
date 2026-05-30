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


def volume_up(step=6):
    _run(f'''
tell application "Spotify"
    set currentVolume to sound volume
    set newVolume to currentVolume + {step}
    if newVolume > 100 then set newVolume to 100
    set sound volume to newVolume
end tell
''')


def volume_down(step=6):
    _run(f'''
tell application "Spotify"
    set currentVolume to sound volume
    set newVolume to currentVolume - {step}
    if newVolume < 0 then set newVolume to 0
    set sound volume to newVolume
end tell
''')
