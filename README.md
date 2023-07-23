# tts_using_ParallelWaveGAN
This notebook provides a demonstration of the realtime Text to speech in English Language using ESPnet2-TTS and ParallelWaveGAN.
<hr>
# Installation

espnet (version 0.10.6): espnet is an end-to-end speech processing toolkit developed by the ESPnet team. It is primarily used for automatic speech recognition (ASR) and text-to-speech (TTS) tasks. The package provides various pre-trained models, tools for data preparation, and implementations of state-of-the-art speech processing algorithms.

parallel_wavegan (version 0.5.4): parallel_wavegan is a Python library used for generating high-quality speech waveforms from mel spectrograms. It leverages generative adversarial networks (GANs) and parallel processing techniques to achieve faster and more efficient waveform generation for text-to-speech systems.

gdown (version 4.4.0): gdown is a Python library that simplifies the process of downloading large files from Google Drive. It is commonly used to download model files or datasets hosted on Google Drive without requiring manual authentication or URL handling.

typeguard (version 2.13.3): typeguard is a Python library used for runtime type checking and validation. It helps ensure that functions and methods are called with the correct argument types and provides better type hinting support.

espnet_model_zoo: This is likely a subpackage or module within the espnet package mentioned earlier. It might contain pre-trained models for automatic speech recognition or text-to-speech tasks that can be used out of the box.

<hr>

# Speaker model demo


# Model Selection
Please select model:


You can try end-to-end text2wav model & combination of text2mel and vocoder.
If you use text2wav model, you do not need to use vocoder (automatically disabled).


Text2wav models:

VITS
Text2mel models:

Tacotron2

Transformer-TTS

(Conformer) FastSpeech

(Conformer) FastSpeech2

Vocoders:

Parallel WaveGAN

Multi-band MelGAN

HiFiGAN

Style MelGAN.

Corpus used

ljspeech_*: LJSpeech dataset

https://keithito.com/LJ-Speech-Dataset/

# About the Speech Corpus used.

The LJSpeech dataset is a widely used and freely available dataset for training and evaluating text-to-speech (TTS) models. It is commonly used as a benchmark dataset in the field of speech synthesis. Here are the key details about the LJSpeech dataset:


Description:


LJSpeech is a collection of text passages from the book "The LJ Speech Dataset" by Keith Ito, where LJ stands for "LibriVox + Jewel." The dataset contains 13,100 audio clips of a single female speaker reading passages from this book. The text includes various sentences, paragraphs, and articles, covering diverse topics. Audio Format:


All audio clips in the dataset are in the WAV format. Each audio clip represents the spoken version of a corresponding text passage. Text Content:


The text data in the LJSpeech dataset is represented as plain text. Each audio clip corresponds to a line of text. The sentences in the text passages are often short and easy to read. Language:


The text in the LJSpeech dataset is primarily in English. Purpose:


The LJSpeech dataset is designed for training and evaluating text-to-speech models, particularly models that convert text into speech in English. Researchers and developers use this dataset to build and test various TTS models. Usage:


The dataset is widely used in the TTS research community to train and benchmark various deep learning-based TTS models. It helps researchers and developers compare the performance of their models and assess the quality of the generated speech. License:


The LJSpeech dataset is released under the Creative Commons Attribution 4.0 International License, which allows users to share and adapt the dataset for any purpose, even commercially, as long as proper attribution is provided to the original author. Availability:


The LJSpeech dataset can be freely downloaded from the following URL: https://keithito.com/LJ-Speech-Dataset/ Overall, the LJSpeech dataset is an essential resource for advancing the field of text-to-speech synthesis and serves as a common benchmark for comparing the performance of various TTS models. Its availability and permissibility have contributed to its widespread use in the research community.

