import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt
from scipy.io.wavfile import read

path = './myvoice.wav'
start = 0
N = 1024

# ファイル読み込み
fs, data = read(path)

# 対数振幅スペクトル
rfft_signal = np.fft.rfft(data[start:N-1+start])
spectrum_abs = np.abs(rfft_signal)
spectrum_log = np.log(spectrum_abs)
frequency = np.arange(N//2) * (fs / N)

# ケプストラム
cn = fftpack.ifft(spectrum_log)

# ローパスフィルタ
index = 40
cn[index:len(cn)-index] = 0
cn_low = np.real(np.fft.fft(cn))

# グラフ描画
fig = plt.figure()
ax2 = fig.add_subplot(111)
ax2.plot(frequency, spectrum_log, c="red")
ax2.plot(frequency, cn_low, marker='|', c="blue")
ax2.set_xlim([0,4000])
ax2.set_xlabel("Frequency [Hz]") 
ax2.set_ylabel("Amplitude [dB]")
plt.show()
plt.close()