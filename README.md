# Video Processing and Subtitle Generation Bot

## Overview

This project is a video processing bot that extracts audio from video files, performs speech recognition, generates subtitles, and allows for text correction and dialect conversion. It utilizes various libraries for audio processing, speech recognition, and translation.

## Features

- **Audio Extraction**: Extracts audio from video files.
- **Speech Recognition**: Converts audio to text using VAD (Voice Activity Detection).
- **Subtitle Generation**: Creates SRT files for subtitles.
- **Text Correction**: Uses OpenAI's API to correct transcription errors.
- **Dialect Conversion**: Converts subtitles to specified dialects.
- **Web Interface**: A Flask-based web interface for user interaction.

## Requirements

- Python 3.x
- Required libraries:
  - `flask`
  - `pyrogram`
  - `librosa`
  - `soundfile`
  - `numpy`
  - `deep_translator`
  - `transformers`
  - `openai`
  - `noisereduce`
  - `pyloudnorm`
  - `av`
  - `werkzeug`
  

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required dependencies.

3. Set up your OpenAI API key in the `make_it_correct.py` file.

4. Create necessary directories for input and output files:

   ```bash
   mkdir -p data/input_videos data/audio_outputs data/text_outputs data/subtitles data/chunks
   ```

## Usage

### Running the Bot

To run the bot, execute the following command:

```bash
python bot.py
```

### Web Interface

To access the web interface, run:

```bash
python app.py
```

Then navigate to `http://localhost:5000` in your web browser.

### Commands

- **/start**: Start the bot and receive instructions.
- **Upload a video**: Send a video file to the bot for processing.

### Processing Steps

1. Upload a video file.
2. Select the original language of the video.
3. Choose the target language for subtitles.
4. Decide if you want to enhance audio quality.
5. Choose whether to correct the subtitles.
6. Select the desired dialect for the subtitles.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [OpenAI](https://openai.com/) for providing the API for text correction.
- [Hugging Face](https://huggingface.co/) for the speech recognition models.
- [Deep Translator](https://pypi.org/project/deep-translator/) for translation capabilities.
