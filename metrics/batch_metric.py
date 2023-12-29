import os
import torchaudio
from torchmetrics.audio import ShortTimeObjectiveIntelligibility, ScaleInvariantSignalNoiseRatio, SignalNoiseRatio, PerceptualEvaluationSpeechQuality

"""
    批量计算SNR和STOI
    Notes: 文件夹下一层即为音频文件,根据enhanced文件名寻找同名clean音频并计算指标
"""


if __name__ == '__main__':
    clean_path = r"D:\Study\AiShell_test\origin"  # clean音频文件夹
    pred_path = r"D:\Study\AiShell_test\output"  # enhanced音频文件夹
    target_sr = 22050  # 统一采样率, 不需要则设为None

    clean_list = os.listdir(clean_path)
    pred_list = os.listdir(pred_path)

    total_snr = 0.0
    total_stoi = 0.0
    count = 0

    for c in clean_list:
        # clean wav is also in the pred list
        if c in pred_list:
            cpath = os.path.join(clean_path, c)
            ppath = os.path.join(pred_path, c)
            # uniform sample rate to {target_sr}
            clean, c_sr = torchaudio.load(cpath)
            clean = torchaudio.transforms.Resample(orig_freq=c_sr, new_freq=target_sr)(clean)  # 重采样
            pred, p_sr = torchaudio.load(ppath)

            # align clean & pred audio length
            if (clean.shape[1] != pred.shape[1]):
                min_len = min(clean.shape[1], pred.shape[1])
                clean = clean[:, 0 : min_len]
                pred = pred[:, 0: min_len]

            assert len(clean) == len(pred), "clean与pred长度不相同, clean: {} - pred: {}".format(len(clean), len(pred))
            assert (c_sr != p_sr), "不匹配的采样率 clean: {} - pred: {}".format(c_sr, p_sr)
            # SNR
            snr_metric = SignalNoiseRatio()
            snr = snr_metric(pred, clean)
            total_snr += snr
            # STOI
            stoi_metric = ShortTimeObjectiveIntelligibility(16000)
            stoi = stoi_metric(pred, clean)
            total_stoi += stoi
            count += 1
            print(f"SNR&STOI of {c} is {snr}-{stoi}")
        else:
            pass

    print('--'*20)
    print("Average SNR: ", total_snr / count)
    print("Average STOI: ", total_stoi / count)

