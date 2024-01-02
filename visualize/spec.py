import os
import torch
import torchaudio
import librosa
import matplotlib.pyplot as plt

"""
    频谱类特征 可视化
"""

def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    """
    @description: 绘制频谱图，通过matplotlib实现
    :param specgram: 频谱矩阵,
    :param title:
    :param ylabel:
    :param ax:
    :return:
    """
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Frame")
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")


if __name__ == '__main__':
    N_FFT = 1024
    WIN_LENGTH = 1024
    HOP_LENGTH = 512
    N_MELS = 128

    fpath = r"../files/clean/p232_208.wav"
    y, sr = torchaudio.load(fpath)
    # 定义 STFT 转换器
    spec_transfer = torchaudio.transforms.Spectrogram(N_FFT, WIN_LENGTH, HOP_LENGTH)
    # 计算 STFT
    spec = spec_transfer(y)
    # 绘制 频谱图
    plot_spectrogram(spec[0], title="spectrogram")
    plt.show()

    # 定义 MelSpectrogram 转换器
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=N_FFT,
        win_length=WIN_LENGTH,
        hop_length=HOP_LENGTH,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        n_mels=N_MELS,
        mel_scale="htk",
    )
    # 计算MelSpectrogram
    mel_spec = mel_spectrogram(y)
    # 绘制mel频谱图
    plot_spectrogram(mel_spec[0], title="MelSpectrogram - torchaudio", ylabel="mel freq")
    plt.show()
