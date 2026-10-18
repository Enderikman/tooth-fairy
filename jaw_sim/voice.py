"""Spoken commands for the simulation, recognised locally with NVIDIA Parakeet TDT 0.6B v3 (onnx-asr).

The listener records short chunks of audio from the system default microphone, recognises them on
the CPU and puts the matching command name into a queue that the simulation drains every frame.
Teeth are spoken as two digits: "vier sechs" / "four six" for tooth 46 (the digits also match when
the recogniser writes them as "4, 6" or "46"). Parakeet v3 understands German and English.
No simulator code in here: the same listener will feed Isaac Sim later.

"stop" goes through the same recognition as every other command, so it is only as fast as one
chunk plus transcription (roughly a second). The hardware stop button stays the fast path for
anything safety relevant.

Usage:
    listener = VoiceListener(commands)
    listener.start()
    ...
    listener.stop()

macOS needs "Microphone" permission for the terminal that runs the sim
(System Settings > Privacy & Security > Microphone).
"""

import queue
import threading
import time
from pathlib import Path

import numpy as np
import onnx_asr
import sounddevice

SAMPLE_RATE = 16_000  # Hz, what the model expects
WINDOW_S = 3.0  # seconds of the latest audio handed to the recogniser (long enough for one phrase)
HOP_S = 1.0  # how often the window is recognised
SILENCE_RMS = 0.01  # windows quieter than this are skipped, which saves a lot of CPU
# Parakeet v3 as int8 ONNX files. If this folder is missing, onnx-asr downloads the same model from Hugging Face.
MODEL_DIR = Path.home() / "local/projects/Diagnostikum/praxiswispr/.cache/models/parakeet-tdt-0.6b-v3-compact-int8"
MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"

# Command name -> spoken forms, lowercase. The order decides which command wins when a phrase of
# one command appears inside the transcript of another: teeth first, then the mode commands.
PHRASES: dict[str, tuple[str, ...]] = {
    "46": ("4 6", "vier sechs", "four six"),
    "43": ("4 3", "vier drei", "four three"),
    "33": ("3 3", "drei drei", "three three"),
    "36": ("3 6", "drei sechs", "three six"),
    "stop": ("stop", "stopp"),
    "park": ("park", "parken"),
    "operate": ("start", "weiter", "operate"),
    "calibrate": ("kalibrieren", "kalibrierung", "calibrate", "calibration"),
}


def phrase_to_command(text: str) -> str | None:
    """Finds the command a transcript asks for.

    Spaces are ignored on both sides of the comparison, so "4 6", "vier sechs" and "sechs und
    vierzig" all match the same phrases as "46" and "sechsundvierzig".

    Args:
        text: Raw transcript of one audio chunk.

    Returns:
        The command name, or None when the text contains no known phrase.
    """
    normalised = "".join(
        character if character.isalnum() or character.isspace() else " "
        for character in text.lower().replace("-", " ")
    )
    spaceless = "".join(normalised.split())
    for command, phrases in PHRASES.items():
        for phrase in phrases:
            if "".join(phrase.split()) in spaceless:
                return command
    return None


class VoiceListener:
    """Turns spoken phrases from the default microphone into command names on a queue."""

    def __init__(self, commands: queue.Queue):
        """Creates the listener.

        Args:
            commands: Queue the recognised command names are put into.
        """
        self.commands = commands
        self.model = None
        self.thread = None
        self.running = False
        self.last_heard = ""  # last recognised command name, for the on-screen text
        self.last_text = ""  # last raw transcript, also for the on-screen text
        self.error = ""  # set when the listening thread gave up

    def start(self) -> None:
        """Loads the recognition model and starts listening in a background thread."""
        if MODEL_DIR.is_dir():
            self.model = onnx_asr.load_model("nemo-conformer-tdt", MODEL_DIR, quantization="int8", providers=["CPUExecutionProvider"])
        else:
            self.model = onnx_asr.load_model(MODEL_NAME, quantization="int8", providers=["CPUExecutionProvider"])
        self.running = True
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        """Stops listening and waits for the background thread to finish."""
        self.running = False
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def _listen(self) -> None:
        """Keeps the last WINDOW_S of microphone audio and recognises it every HOP_S.

        The microphone fills `blocks` from its own callback, so recognition (which takes a good
        part of a second) never drops audio. After a command is recognised the audio is thrown
        away, so one spoken phrase gives one command.

        Any failure (no microphone, model trouble) ends the thread and is kept in self.error, so a
        broken microphone never takes the simulation down with it.
        """
        blocks: list[np.ndarray] = []
        window = int(WINDOW_S * SAMPLE_RATE)

        def on_audio(block: np.ndarray, frames: int, timing, status) -> None:
            blocks.append(block[:, 0].copy())

        try:
            with sounddevice.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=on_audio):
                while self.running:
                    time.sleep(HOP_S)
                    if not blocks:
                        continue
                    audio = np.concatenate(blocks)[-window:]
                    blocks[:] = [audio]
                    if float(np.sqrt(np.mean(np.square(audio)))) < SILENCE_RMS:
                        continue
                    self.last_text = self._transcribe(audio)
                    command = phrase_to_command(self.last_text)
                    if command is not None:
                        self.last_heard = command
                        self.commands.put(command)
                        blocks.clear()
        except Exception as exception:  # noqa: BLE001 - whatever breaks, only the voice input dies
            self.error = str(exception)
            self.running = False

    def _transcribe(self, audio: np.ndarray) -> str:
        """Recognises one chunk of audio.

        Args:
            audio: Mono float32 samples at SAMPLE_RATE.

        Returns:
            The spoken text, empty when the chunk held no speech.
        """
        return self.model.recognize(audio, sample_rate=SAMPLE_RATE).strip()
