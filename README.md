# Text-to-speech synthesis using ParallelWaveGAN
This notebook provides a demonstration of the Text to speech in English Language using ESPnet2-TTS and ParallelWaveGAN.
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

# Choose English model

using a user interface (UI) to select an English model for text-to-speech synthesis with the LJSpeech dataset. Here's a breakdown of the choices you have:

Language: English

# Text-to-Mel Models (Choose one):

kan-bayashi/ljspeech_tacotron2: Tacotron 2 - A widely used sequence-to-sequence model for text-to-speech synthesis.

kan-bayashi/ljspeech_fastspeech: FastSpeech - A fast and efficient model for text-to-speech synthesis using feed-forward networks.

kan-bayashi/ljspeech_fastspeech2: FastSpeech 2 - An improved version of FastSpeech, known for its fast synthesis and parallel processing capabilities.

kan-bayashi/ljspeech_conformer_fastspeech2: Conformer FastSpeech 2 - A variant of FastSpeech 2 that uses the Conformer architecture for improved performance.

# Vocoder Models (Choose one):

none: If you choose this, no vocoder will be used. You would have to handle the waveform generation separately.

parallel_wavegan/ljspeech_parallel_wavegan.v1: Parallel WaveGAN - A vocoder that generates high-quality waveforms in parallel using a generative adversarial network.

parallel_wavegan/ljspeech_full_band_melgan.v2: Full Band MelGAN - A mel-spectrogram-based vocoder that produces high-fidelity waveforms.

parallel_wavegan/ljspeech_multi_band_melgan.v2: Multi-Band MelGAN - A variant of MelGAN with multiple frequency bands for better waveform quality.

parallel_wavegan/ljspeech_hifigan.v1: HiFi-GAN - A high-fidelity generative adversarial network for vocoding.

parallel_wavegan/ljspeech_style_melgan.v1: Style MelGAN - A MelGAN variant that can control the voice style in the generated speech. Once you make your selections for the text-to-mel model (tag) and vocoder model (vocoder_tag), you can proceed with text-to-speech synthesis using the chosen models and the LJSpeech dataset in the English language.

<hr>

# Model Setup

using the ESPnet toolkit to perform text-to-speech synthesis with the chosen model and vocoder. Here's a breakdown of the code and its parameters:

espnet2.bin.tts_inference.Text2Speech: This is the class used for text-to-speech synthesis in ESPnet.

Text2Speech.from_pretrained: This method creates a Text2Speech instance from a pre-trained model, allowing you to use the selected text-to-mel model (tag) and vocoder model (vocoder_tag) for speech synthesis.

Parameters:

model_tag: The tag of the selected text-to-mel model.

vocoder_tag: The tag of the selected vocoder model.

device: Specifies the device to run the inference. In this case, "cuda" indicates using a GPU for faster processing if available. If you don't have a GPU, you can use "cpu" instead.

Note: The following parameters are specific to certain text-to-mel models, and some of them may not apply to the chosen model. You may need to adjust them accordingly or omit them if they are not applicable:

threshold: Only used for Tacotron 2 and Transformer models. It determines the threshold for attention constraint during synthesis.

minlenratio: Only used for Tacotron 2. It controls the minimum output length ratio during synthesis.

maxlenratio: Only used for Tacotron 2. It controls the maximum output length ratio during synthesis.

use_att_constraint: Only used for Tacotron 2. If set to True, attention constraint is applied during synthesis.

backward_window: Only used for Tacotron 2. It sets the size of the backward window for attention constraint.

forward_window: Only used for Tacotron 2. It sets the size of the forward window for attention constraint.

speed_control_alpha: Used for FastSpeech, FastSpeech2, and VITS models. It controls the speed of speech synthesis by scaling the durations.

noise_scale: Only used for VITS. It controls the amount of noise to be injected into the input features during inference.

noise_scale_dur: Only used for VITS. It controls the amount of noise to be injected into the duration predictor during inference. With this configuration, you can now use the text2speech instance to perform text-to-speech synthesis on the input text using the selected pre-trained model and vocoder.

<hr>

# Synthesis

Input Sentence Prompt: The code prompts you to input your favorite sentence in the specified language (English in this case).

Text-to-Speech Synthesis:

The input sentence x is passed to the text2speech model, and it generates the corresponding speech waveform (wav) for the input text. The speech synthesis is done using the text2speech instance with the previously selected model and vocoder.

The runtime (RTF - Real-Time Factor) of the synthesis process is calculated, indicating the time taken for the model to synthesize one second of audio. A lower RTF indicates faster synthesis.

Audio Playback:

The synthesized audio waveform (wav) is played back so that you can listen to the generated speech. The audio playback uses IPython's Audio class to display and play the audio in the notebook.

<hr>

# Note

For the synthesis to work, make sure you have selected the appropriate text-to-mel model (tag) and vocoder model (vocoder_tag) during the setup. Also, ensure that the necessary dependencies and pre-trained models are correctly installed and available.

When you run this code and input your favorite sentence, the text-to-speech model will generate the corresponding speech, and you'll be able to listen to the generated audio. The RTF value will give you an idea of the synthesis speed compared to real-time. Lower RTF values indicate faster synthesis.

<hr>

# Output Links:

https://drive.google.com/drive/folders/1VVyUrZ0Vsyx2by79esMpjtQFwhTwiJxT?usp=sharing



