## Blur Metrics

In our paper, we combined 5 blur metrics to evaluate the quality of the acquired pCLE images:
* Mean of Intensity (MoI)
* Non-Reference Blur Metric (NFBM)
* Fast Fourier Transform (FFT)
* Variance of Laplace Distribution (LAPV)
* Gaussian Derivative (GDER)

The final BM value is achieved by the averaging 5 metrics:

${\rm BM} = \frac{\rm MoI + NFBM + FFT + LAPV + GDER}{5}$

## How to generate

The `blur_metric.py` should be put in the root folder of the dataset.

The data folder structure should be:
```
root folder
- - - - - pCLE_Video_1.mkt
- - - - - - - - - - 1.pkl
- - - - - - - - - - 2.pkl
- - - - - - - - - - 3.pkl
......
- - - - - pCLE_Video_2.mkt
- - - - - - - - - - 1.pkl
- - - - - - - - - - 2.pkl
- - - - - - - - - - 3.pkl
......
- - - - - pCLE_Video_3.mkt
- - - - - - - - - - 1.pkl
- - - - - - - - - - 2.pkl
- - - - - - - - - - 3.pkl
......
```
The `blur_metric.py` can generate the blur metric value for each frames in each pCLE video folder by running:
```
python3 blur_metric.py
```