**phasefinder** is a beat estimation model that predicts metric position as rotational phase, heavily inspired by [this paper](https://archives.ismir.net/ismir2021/paper/000061.pdf).


[full writeup here](https://bleu.green/phasefinder)


# usage

## python 

```python
from phasefinder import Phasefinder

# initialize model
pf = Phasefinder()

# predict beats
audio_path = "path/to/your/audio/file.[wav/mp3/flac/etc]"
beat_times = pf.predict(audio_path)

# predict beats and BPM
beat_times, bpm = pf.predict(audio_path, include_bpm=True)

# generate a click track
output_path = "output_with_clicks.wav"
pf.make_click_track(audio_path, output_path, beats=beat_times)
```

## cli

### basic usage

```bash
python -m phasefinder.infer path/to/your/audio/file.wav
```

This will print the estimated beat times to the console.

### options

- `--bpm`: Include BPM in the output
- `--noclean`: Don't apply the cleaning function to the beat times
- `--format {times,click_track}`: Choose the output format (default: times)
  - `times`: Output beat times
  - `click_track`: Generate an audio file with click track
- `--audio_output PATH`: Specify the path for the output audio file with clicks (default: output_with_clicks.wav)
- `--json_output PATH`: Save the results to a JSON file

### examples

1. Estimate beats and BPM:
   ```bash
   python -m phasefinder.infer path/to/audio.wav --bpm
   ```

2. Generate a click track:
   ```bash
   python -m phasefinder.infer path/to/audio.wav --format click_track --audio_output output.wav
   ```

3. Save results to a JSON file:
   ```bash
   python -m phasefinder.infer path/to/audio.wav --bpm --json_output results.json
   ```

4. Estimate beats without applying the cleaning function:
   ```bash
   python -m phasefinder.infer path/to/audio.wav --noclean
   ```