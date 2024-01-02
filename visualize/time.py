import torch
import torchaudio
import os
import librosa
import matplotlib.pyplot as plt

def plot_waveform(waveform, sr, title="Waveform", ax=None):
    """
    @discription: 绘制波形，通过matplotlib
    :param waveform: 波形, 支持tensor
    :param sr: 采样率
    :param title:
    :param ax:
    :return:
    """
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)
    ax.set_xlabel("Time/s")


if __name__ == '__main__':
    fpath = r"../files/clean/p232_208.wav"
    y, sr = torchaudio.load(fpath)
    plot_waveform(y, sr)
    plt.show()
