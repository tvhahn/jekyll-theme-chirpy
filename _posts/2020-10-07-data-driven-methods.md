---
title: "Data-Driven Methods - Advances in Condition Monitoring, Pt II"
date: "2020-10-07 18:15"
categories: [Condition Monitoring, Data-Driven Methods]
tags: [manufacturing, ai, machine learning, condition monitoring, feature engineering, end-to-end deep learning, deep learning, tool wear]     # TAG names should always be lowercase
description: "In part two, we give an  overview of the two data-driven approaches in condition monitoring; that is, feature engineering and end-to-end deep learning. Which approach should you use? Well, it all depends..."
---

<div style="text-align: center; ">
<figure>
  <img src="/assets/img/drill.jpg" alt="milling tool" style="background:none; border:none; box-shadow:none; text-align:center"/>
</figure>
</div>

> In part two, we give an  overview of the two data-driven approaches in condition monitoring; that is, feature engineering and end-to-end deep learning. Which approach should you use? Well, it all depends...

Imagine you're the CEO of a manufacturing company. You make small components out of metal, manufactured on CNC machines. One important jobs you have is to improve the productivity of your operation. If you don't, well, you'll slowly see your company's profits eaten away by the competition. Not good. What can you do as an enterprising CEO to prevent this? Use data-driven condition monitoring methods! You can use those methods to predict when a CNC tool is worn and needs replacing. Ultimately, it will free up the time of your operators and reduce waste.

There are two dominant data-driven approaches in condition monitoring; that is, feature engineering, the classical approach, and end-to-end deep learning, the newer approach. As CEO, you and your underlings will have to decide on which approach to use. We'll be exploring these approaches, along with things to consider, and a bit of my opinion, in the remainder of this post.

# Feature Engineering Approach

Condition monitoring involves the collection of numerous parameters -- such as vibration waveforms, electrical current signals, speed signals, etc. -- in order to detect faults in machinery. But what happens after these signals are collected? In feature engineering, an expert uses the signals to design features. The features will ideally contain "information" that is indicative of the health of the machinery. The best features can then be used by machine learning models to make predictions on the health of the machinery. The figure below shows the complete feature engineering approach, from collecting the data to using the final model for prediction.


<div style="text-align: left;">
<figure>
  <img src="/assets/img/feature_engineering_approach.svg" alt="feature engineering approach" style="background:none; border:none; box-shadow:none; text-align:center"/>
  <figcaption style="color:grey; font-size:smaller"><br>The feature engineering approach relies on manually designed features (signal processing or statistical features are common in condition monitoring). Following feature selection, various machine learning models can be used to make predictions.</figcaption>
</figure>
</div>


There are a wide variety of methods for designing features, but the creation of statistical features is the simplest approach. In this method, statitics, such as root-mean-square, kurtosis, or standard deviation, to name a few, are calculated directly on the signals from the machinery.

Many advanced feature design methods come from the field of [signal processing](https://en.wikipedia.org/wiki/Signal_processing), and the [Fast Fourier Transform](https://en.wikipedia.org/wiki/Fast_Fourier_transform) (FFT) is one such method. The FFT transforms the signal from the time domain to the frequency domain. Useful features can then be extracted from the frequency domain, such as the peak freaquency, the average value, or the natural frequency. The process of applying a FFT on a current signal is demonstrated in the code snippet below. And if you want a wonderful explanation of the Fourier transform you must watch [3Blue1Brown's video](https://youtu.be/spUNpyF58BY) on Youtube.


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tvhahn/blog-snips/blob/main/2020.10.07_data_driven_methods/fft.ipynb)
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal, fftpack
import seaborn as sns

# load the current data from a csv
df = pd.read_csv("current.csv", index_col=False)
y = df["current"].to_numpy(dtype="float64")  # convert to a numpy array

# setup the seaborn plot
sns.set(font_scale=1.0, style="whitegrid")
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False, sharey=False, dpi=600)
fig.tight_layout(pad=5.0)

pal = sns.cubehelix_palette(6, rot=-0.25, light=0.7)  # pick nice color for plot

# parameters for plot
T = 1.0 / 1000.0  # sample spacing
N = len(y)  # number of sample points
x = np.linspace(0.0, N * T, N)

# plot time domain signal
axes[0].plot(x, y, marker="", label="Best model", color=pal[3], linewidth=0.8)
axes[0].set_title("Time Domain", fontdict={"fontweight": "normal"})
axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("Current")
axes[0].set_yticklabels([])

# do some preprocessing of the current signal
y = signal.detrend(y, type == "constant")  # detrended signal
y *= np.hamming(N)  # apply a hamming window. Why? https://dsp.stackexchange.com/a/11323

# FFT on time domain signal
yf = fftpack.rfft(y)
yf = 2.0 / N * np.abs(yf[: int(N / 2.0)])
xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

# plot the frequency domain signal
axes[1].plot(xf, yf, marker="", label="Best model", color=pal[3], linewidth=0.8)
axes[1].set_title("Frequency Domain", fontdict={"fontweight": "normal"})
axes[1].set_xlabel("Frequency (Hz)")
axes[1].set_ylabel("Current Magnitude")

# clean up the sub-plots to make everything pretty
for ax in axes.flatten():
    ax.yaxis.set_tick_params(labelleft=True, which="major")
    ax.grid(False)

plt.show()
```

<div style="text-align: center;">
<figure>
  <img src="/assets/img/time_freq_domains.svg" alt="time domain versus frequency domain for a current signal" style="background:none; border:none; box-shadow:none; text-align:center"/>
  <figcaption style="color:grey; font-size:smaller"><br>Time domain versus frequency domain for a current signal.</figcaption>
</figure>
</div>


After feature design, the best features can be selected using a number of techniques. [Pricipal component analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) (PCA) can be used to represent the features in a lower dimensional space that hopefully including the most useful information. A random search can be used to randomly features for model building. Or you can select the features that have the least correlation between each other using the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient). There are many other techniques, and we'll  have to explore some of these methods in future posts...

The next step in the process is model training. Depending on your methodology, you may train several types of machine learning models and see which one performs best. And as always, data science best practices must be used, such as splitting your data set into a training, validation, and testing sets, or using k-fold cross-validation.

Finally, once you've trained your models and selected the best one, you can use it for prediction. You now have a model that you can deploy in your manufacturing process!

# End-to-End Deep Learning Approach
A deep learning system is a neural network with many layers. Neural networks, trained with [backpropagation](https://en.wikipedia.org/wiki/Backpropagation), have been used since the 1980s.[^rumelhart1985learning] However, not until the early 2010s, with the exponential increase in computational power and large data sets, did neural networks, stacked in many layers (hence the "deep" in deep learning), show their full potential. These deep neural networks can learn complex non-linear relationships in the data that are unobservable to humans.

Manually designing features can be time-consuming. Yet, this was the primary method of creating data-driven condition monitoring applications before the emergence of deep learning. In the textbook [Deep Learning](https://www.deeplearningbook.org/), the authors note how an entire research community can spend decades developing the best manually designed features only to have a deep learning approach achieve superior results in a matter of months, as was the case in computer vision and natural language processing. As a result, deep learning methods are of increased interest for condition monitoring applicatoins.[^zhao2019deep] The focus has changed from designing features to designing neural network architectures.

Many of the deep learning applications within condition monitoring utilize a data preprocessing step. This is done to make the "learning" easier for the neural network. Often, the preprocessing step involves transforming the data from the time domain to the frequency domain, or by using some other advanced feature engineering techniques to extract useful information. However, advances in other fields, such speech recognition, have shown that deep learning networks can work directly on raw waveforms in an end-to-end fashion. As a result, much data-driven condition monitoring research now uses this end-to-end deep learning approach, which is illustrated below.

<div style="text-align: left;">
<figure>
  <img src="/assets/img/end_to_end_deep_learning.svg" alt="end-to-end deep learning approach" style="background:none; border:none; box-shadow:none; text-align:center"/>
  <figcaption style="color:grey; font-size:smaller"><br>The end-to-end deep learning approach works directly on the raw signals from the monitored machine, thus removing the need for feature engineering.</figcaption>
</figure>
</div>


# Advantages and Disadvantages

Now that we have a better understanding of the two approaches, which ones should you use? It all depends, as each approach has its advantages and disadvantages. The below graphic gives a quick summary of the pros and cons for each approach.


<div style="text-align: center;">
<figure>
  <img src="/assets/img/pro_con.svg" alt="pros and cons for feature engineering and end-to-end deep learning" style="background:none; border:none; box-shadow:none; text-align:center" width="500px"/>
  <figcaption style="color:grey; font-size:smaller"><br>The pros and cons for feature engineering and end-to-end deep learning.</figcaption>
</figure>
</div>

Feature engineering can be extremely labor intensive, and that is its biggest drawback. However, there can be benefits to using this approach in industrial applications. The decisions made by models that use feature engineering are often more interpretable than those decisions made from the end-to-end deep learning approach. As an example of this interpretability, a data scientists can easily understand which features were most useful to a random forest model. However, an end-to-end deep learning model creates features through its many layers of neurons, and the complex non-linearities in the neural network makes the interpretability of the network challenging. Understanding how a model makes its predictions can be beneficial when those predictions carry significant economic impact, as can be the case in industrial and manufacturing environments. 

In addition to the interpretability, the feature engineering approach can generally operate on less data and computational power than an end-to-end deep learning approach. A simple linear model may require a few parameters and yield acceptable results for many different applications. Thus, less data will be needed to train the model -- hundreds to thousands of data samples. However, an end-to-end deep learning model may include millions of parameters, and as a result much more data will be required to train the model -- tens-of-thousands to millions of data samples.

# Opinion

Feature engineering leans into the expertise of the human. End-to-end deep learning -- and deep learning in general -- leans into the data and computational power present in today's world. The need for large amounts of data can be a drawback for the end-to-end deep learning approach. But from my personal experience, collecting enough data in industrial and manufacturing environments is *not* the constraint. Rather, it's the temptation to use the less general feature engineering approach, as opposed to the more general end-to-end deep learning approach. 

Feature engineering doesn't require as much data, and one can seemingly iterate through ideas quicker, providing a satisfying taste of "progress". Deep learning, is opaque and is more reliant on having a robust infrastructure setup -- requiring a capital outlay -- to easily collect and label new data. However, the dominance of deep learning techniques in other fields, such as computer vision and speech recognition, should be a harbinger of things to come in the field of condition monitoring. General methods, powered by the exponential rise of computational power, seems to win-out each time when put up against the efforts of domain experts. Richard Sutton, a grandfather of reinforcement learning, calls this the "[bitter lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)".

# Conclusion
We've reviewed the two dominant data-driven approaches in condition monitoring. Feature engineering relies on an expert to manually design features, which can then be used by machine learning models to make decisions and predictions. End-to-end deep learning removes the need for advanced feature engineering expertise. Instead it relies on a deep neural network, the data, and computational power, to understand complicated problems. There are times to use feature engineering (I'll explore some applications in a future post) and times to use deep learning. But if one thing can be said, it's this: we live in a time of abundant computational power. Embrace it.

# References
[^rumelhart1985learning]: Rumelhart, David E., Geoffrey E. Hinton, and Ronald J. Williams. *[Learning internal representations by error propagation](https://apps.dtic.mil/dtic/tr/fulltext/u2/a164453.pdf)*. No. ICS-8506. California Univ San Diego La Jolla Inst for Cognitive Science, 1985. 

[^zhao2019deep]: Zhao, Rui, et al. "[Deep learning and its applications to machine health monitoring](https://doi.org/10.1016/j.ymssp.2018.05.050)." *Mechanical Systems and Signal Processing* 115 (2019): 213-237.
