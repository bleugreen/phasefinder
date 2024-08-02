
import librosa
import numpy as np
import soundfile as sf
import argparse
import json

from phasefinder.predictor import Phasefinder

if __name__ == '__main__':
    pf = Phasefinder()

    parser = argparse.ArgumentParser(description='Predict beats from an audio file.')
    parser.add_argument('audio_path', type=str, help='Path to the audio file')
    parser.add_argument('--bpm', action='store_true', help='Include BPM in the output')
    parser.add_argument('--noclean', action='store_true', help='Don\'t apply cleaning function')
    parser.add_argument('--format', type=str, choices=['times', 'click_track'], default='times', help='Output format: "times" for beat times or "click_track" for audio with click track')
    parser.add_argument('--audio_output', type=str, default='output_with_clicks.wav', help='Path to save the output audio file with clicks')
    parser.add_argument('--json_output', type=str, default='', help='Path to save the output json results')

    args = parser.parse_args()

    audio_path = args.audio_path
    if args.bpm:
        beat_times, bpm = pf.predict(audio_path, include_bpm=args.bpm, clean=not args.noclean)
    else:
        beat_times = pf.predict(audio_path, include_bpm=args.bpm, clean=not args.noclean)

    if args.format == 'click_track':
        audio, sr = librosa.load(audio_path)
        click_track = librosa.clicks(times=beat_times, sr=sr, length=len(audio))
        audio_with_clicks = np.array([click_track, audio])
        audio_with_clicks = np.vstack([click_track, audio]).T
        sf.write(args.audio_output, audio_with_clicks, sr)
    else:
        if args.json_output != '':
            output_data = {
                'beat_times': beat_times.tolist()
            }
            if args.bpm:
                output_data['bpm'] = bpm

            with open(args.json_output, 'w') as json_file:
                json.dump(output_data, json_file, indent=4)
        else:
            print(f"beats = {beat_times}")
            if args.bpm:
                print(f'bpm = {bpm}')
