import librosa
import torchaudio
from torchmetrics.audio import ShortTimeObjectiveIntelligibility, ScaleInvariantSignalNoiseRatio, SignalNoiseRatio, PerceptualEvaluationSpeechQuality

'''
    计算常用语音指标——基于torchaudio
'''

if __name__ == '__main__':
    clean_path = r"..\files\clean\p232_208.wav"
    pred_path = r"..\files\pred\p232_208.wav"

    # 统一采样率
    clean, sr = torchaudio.load(clean_path)
    clean = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(clean)
    print("clean-sr: ", clean.shape, sr)
    pred, sr = torchaudio.load(pred_path)
    print("pred-sr: ", pred.shape, sr)

    # 计算SNR
    snr_metric = SignalNoiseRatio()
    snr = snr_metric(pred, clean)
    print("snr: ", snr)

    # 计算STOI
    stoi_metric = ShortTimeObjectiveIntelligibility(16000)
    stoi = stoi_metric(pred, clean)
    print("stoi: ", stoi)

    # 计算PESQ
    pesq_nb_metric = PerceptualEvaluationSpeechQuality(16000, 'nb')  # 窄带
    pesq_wb_metric = PerceptualEvaluationSpeechQuality(16000, 'wb')  # 宽带
    pesq_nb = pesq_nb_metric(pred, clean)
    pesq_wb = pesq_wb_metric(pred, clean)

    print('pesq_nb: ', pesq_nb)
    print('pesq_wb: ', pesq_wb)

