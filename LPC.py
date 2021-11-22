import soundfile as sf
import librosa 
import math
import scipy
import numpy as np
from scipy.signal import hamming
import scipy.signal as signal
from scipy.io.wavfile import write

import matplotlib.pyplot as plt
data, sam_freq = sf.read(r"C:\Users\91931\OneDrive\Desktop\PP 2021\lab last\phonecalls.wav")
print(len(data))
print(data.shape)
#print(data)
#print(data, fs)


#Separate the frames
def separate_frame(signal, window, r = 0.5):
    #print(signal)
    #print(window)
    len_sig = len(signal)
    len_window = len(window)
    app_len = math.floor((1 - r)*len_window)
    diff_len = math.floor((len_sig - len_window) / app_len) + 1
    frame = np.zeros((diff_len, len_window))
    for i in range(diff_len):
        n = i * app_len
        frame[i, :] = window * signal[n : len_window + n]
    return frame

wind =hamming(math.floor(0.04*sam_freq))

#print(wind)
frames = separate_frame(data, wind)
#print(len(frames))
#print(len(frames[0])))


#Funnction to plot LPC and DFT
def plot_lpc_dft(frames, frame_no, order):
    frame = frames[frame_no]
    l = int(len(frame)/2)
    
    #DFT plot
    dft_frame = np.fft.fft(frame)
    x = np.arange(len(frame)/2)
    maximum = max(x)
    y1 = np.log10(abs(dft_frame[:l]))
   
    plt.plot(x/maximum,y1)
    
    #LPC plot
    lpc_coeff = librosa.core.lpc(frame,order)
    w,h = scipy.signal.freqz([0.008],lpc_coeff,worN=l)
    
    plt.plot(w/np.pi,np.log10(abs(h)))
    
    
    plt.legend(["DFT","LPC"])
    plt.title('DFT and LPC Spectrum of frame: '+str(frame_no)+' and order: '+str(order))
    plt.xlabel('frequency')
    plt.ylabel('log10|Y|')
    plt.show()
    
plot_lpc_dft(frames,30,16)   
    
#synthesize the frame
def Synthetic(frame, order):
        poles = 1
        lpc_coeff = librosa.core.lpc(frame,order)
        residual = signal.lfilter(lpc_coeff, poles, frame)
        synthetic = signal.lfilter([1], lpc_coeff,residual)
    
        return synthetic
    
#adding the synthesize frames
def Add_Frames(frames,window,r=0.5):
    total_count, len_frame = frames.shape
    app_len = np.floor(len_frame*r)
    num = (total_count - 1) * app_len + len_frame
    add_frame = np.zeros((int(num), ))
    order = 8
    for i in range(total_count):
        n = int(i*app_len)
        add_frame[n:len_frame+n] += Synthetic(frames[i,:],order)
    return add_frame


synthesize_data = Add_Frames(frames, 15)
write(r"C:\Users\91931\OneDrive\Desktop\PP 2021\lab last\kar6.wav", sam_freq, synthesize_data)

