import os
import numpy as np
import glob
from scipy.ndimage import filters, laplace

def MoI(img):
    return np.mean(img)


def NFBM(img):
    avg_hor = np.ones((1,9)) / 9
    avg_ver = avg_hor.T
    B_hor = filters.convolve(img, avg_hor)
    B_ver = filters.convolve(img, avg_ver)
    D_F_ver = np.abs(img[:, :-1] - img[:, 1:])
    D_F_hor = np.abs(img[:-1, :] - img[1:, :])
    D_B_ver = np.abs(B_ver[:, :-1] - B_ver[:, 1:])
    D_B_hor = np.abs(B_hor[:-1, :] - B_hor[1:, :])
    T_ver = D_F_ver - D_B_ver
    T_hor = D_F_hor - D_B_hor
    V_ver = np.maximum(T_ver, 0)
    V_hor = np.maximum(T_hor, 0)
    S_F_ver = np.sum(D_F_ver[1:-1, 1:-1])
    S_F_hor = np.sum(D_F_hor[1:-1, 1:-1])
    S_V_ver = np.sum(V_ver[1:-1, 1:-1])
    S_V_hor = np.sum(V_hor[1:-1, 1:-1])
    B_ver = (S_F_ver - S_V_ver) / S_F_ver
    B_hor = (S_F_hor - S_V_hor) / S_F_hor
    blur = np.maximum(B_hor, B_ver)
    return blur

def FFT(img):
    h, w = img.shape
    ws = 16
    CM = h // 2
    CN = w // 2
    im_fft = np.fft.fft2(img)
    im_fft_shift = np.fft.fftshift(im_fft)
    im_fft_shift[CM-ws:CM+ws, CN-ws:CN+ws] = 0
    im_fft_ishift = np.fft.ifftshift(im_fft_shift)
    im_ifft = np.fft.ifft2(im_fft_ishift)
    magnitude = 20*np.log(np.abs(im_ifft))
    blur = np.mean(magnitude)
    return blur

def LAPV(img):
    im_lap = laplace(img)

    return np.std(im_lap)**2

def GDER(img):
    N = 15 // 2
    sig = N / 2.5
    a = np.arange(-N, N+1)
    x, y = np.meshgrid(a, a)
    G = np.exp(-(x**2 + y**2) / (2 * sig**2)) / (2*np.pi*sig)
    Gx = -x * G / (sig**2)
    Gx = Gx / (np.sum(Gx)+1e-7)
    Gy = -y * G / (sig**2)
    Gy = Gy / (np.sum(Gy)+1e-7)
    Rx = filters.convolve(img, Gx)
    Ry = filters.convolve(img, Gy)
    FM = Rx**2 + Ry**2

    return np.mean(FM)
    
root_path = "./" 
video_dir = sorted(glob.glob(root_path + "/*.mkt"))
for folder in video_dir:
    video_path = os.path.join(folder, "video.npy")
    video = np.load(video_path)
    num = video.shape[0]
    blur1 = np.zeros(num)
    blur2 = np.zeros(num)
    blur3 = np.zeros(num)
    blur4 = np.zeros(num)
    blur5 = np.zeros(num)
    for i in range(num):
        frame = video[i]
        blur1[i] = MoI(frame)
        blur2[i] = NFBM(frame)
        blur3[i] = FFT(frame)
        blur4[i] = LAPV(frame)
        blur5[i] = GDER(frame)
    
    
    blur1_norm = (blur1 - np.min(blur1)) / (np.max(blur1) - np.min(blur1))
    blur2_norm = (blur2 - np.min(blur2)) / (np.max(blur2) - np.min(blur2))
    blur3_norm = (blur3 - np.min(blur3)) / (np.max(blur3) - np.min(blur3))
    blur4_norm = (blur4 - np.min(blur4)) / (np.max(blur4) - np.min(blur4))
    blur5_norm = (blur5 - np.min(blur5)) / (np.max(blur5) - np.min(blur5))
    avg_blur_norm = (blur1_norm + blur2_norm + blur3_norm + blur4_norm + blur5_norm) / 5
    np.save(os.path.join(folder, "BM.npy"), avg_blur_norm)
    print(folder)
print("DONE!")



    