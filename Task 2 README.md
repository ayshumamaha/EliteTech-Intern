# Speech Recognizer
## Overview
This project is a simple **Speech Recognition** program built using Python and the `SpeechRecognition` library. It takes a WAV audio file as input and converts the spoken content into text using Google's Speech Recognition API.

## Features
* Converts speech from an audio file into text.
* Supports `.wav` audio files.
* Uses Google's speech recognition service for transcription.
* Handles common errors such as unclear speech, service request errors, and missing audio files.
* Includes a sample usage section for testing the program.

## Requirements
Install the required Python library using:
The program uses the `speech_recognition` module to read and process the audio file.

## How It Works
The `transcribe_audio()` function receives the path of a WAV audio file. A `Recognizer` object is created to process the audio. The program reads the audio using `AudioFile`, records the audio data, and sends it to Google's speech recognition service.
If the speech cannot be understood, the program returns an appropriate error message. It also handles situations where the Google service cannot be reached or the specified audio file does not exist.

## Usage
Place your WAV audio file in the appropriate location and update the file path in the program:
The transcribed text will be displayed in the terminal.

## Error Handling
The program handles:
* Unclear or unrecognizable speech.
* Problems connecting to the Google Speech Recognition service.
* Missing audio files.

Author
M. Ayshwarya
