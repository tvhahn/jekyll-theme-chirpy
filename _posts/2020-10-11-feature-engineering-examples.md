---
title: "The Case for Feature Engineering - Advances in Condition Monitoring, Pt III" 
date: "2020-10-11 10:40" 
categories: [Condition Monitoring, Feature Engineering] 
tags: [ai, machine learning, condition monitoring, feature engineering, total harmonic distortion, signal processing, internal combustion engine, logistic regression, mining industry, mobile equipment]      
description: "Feature engineering can be powerful tool, but it doesn't have to be complicated. Often, the simple solution is best."
image:
math: true
--- 

> Feature engineering can be powerful tool, but it doesn't have to be complicated. Often, the simple solution is best.

In part two of this series, we went over the differences between the feature engineering approach and the end-to-end deep learning approach. The feature engineering approach can be labor intensive. But, I don't want to discourage you. Feature engineering can be a great choice for many applications within the condition monitoring space. In fact, if you have a problem that can be elegantly solved using straightforward feature design and a simple model, then you should use that first. As noted by Daniel Kahneman in his book *[Thinking Fast and Slow](https://www.amazon.com/dp/0374533555/ref=cm_sw_r_tw_dp_x_dJFGFb0YHDJEV)*, simple statistical rules regularly outperform more complex models. What would some of these simple applications look like? Here are a couple of examples. 

# Frequency Domain Technique for Combustion Engines

<div style="text-align: center; ">
<figure>
  <img src="/assets/img/engine.jpg" alt="engine" style="background:none; border:none; box-shadow:none; text-align:center"/>
  <figcaption style="color:grey; font-size:smaller">Photo by <a href="https://www.pexels.com/@phong-nguyen-40565?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels">Phong Nguyen</a></figcaption>
</figure>
</div>

The first application we'll look at is more of a feature engineering technique. In this example, [total harmonic distortion](https://en.wikipedia.org/wiki/Total_harmonic_distortion) (THD) is used to detect abnormalities on reciprocating machinery -- machinery like internal combustion engines or reciprocating compressors. The application comes from a [patent](https://patents.google.com/patent/US9791343B2/en) by Jeffrey Bizub of GE.[^bizub2017methods] I don't usually appreciate reading patents, but this one was clear, and I like the simplicity of this idea. 

The patent describes how an internal combustion engine produces a characteristic vibration pattern while in operation. The vibration signal is a superposition of many waveforms. Each component in the engine will have a primary vibration waveform (called its fundamental waveform) and then subsequent smaller waveforms belonging to that component (called harmonic waveforms).

When looking in the frequency domain, the fundamental waveform for a specific component will be represented by a large peak, and the harmonic waveforms will be represented by smaller peaks. The fundamental waveform vibrates at the fundamental frequency (or the natural frequency). The harmonic waveforms vibrate at frequencies that are at integral multiples of the fundamental frequency. The figure below, taken from the patent, shows a frequency domain representation of a signal, with the fundamental and harmonic frequencies labelled.

<div style="text-align: center; ">
<figure>
  <img src="/assets/img/harmonics.png" alt="fundamental and harmonic peaks on a frequency domain signal" style="background:none; border:none; box-shadow:none; text-align:center"/>
  <figcaption style="color:grey; font-size:smaller">The fundamental and harmonic frequencies. Figure from the patent by Jeffrey Bizub.</figcaption>
</figure>
</div>

The vibration signals will distort as components in the engine wear and the harmonics of these components become more dominant. This change in the waveform of the component can be measured by the total harmonic distortion, expressed as: 

$$\text{THD} = \frac{\sqrt{(f_2^2+f_3^2+f_4^2+ \cdots + f_n^2)}}{f_1} \times 100 $$

where $$f_2$$, $$f_3$$, ... $$f_n$$ are the amplitudes of the $$n$$ harmonic frequencies, and $$f_1$$ is the amplitude of the fundamental frequency.

The THD value can be trended over time, and a threshold value set to indicate when the engine component is in an unhealthy condition. The THD feature is used in other data-driven condition monitoring applications as well, from detecting faults in electric motors[^f1993ieee], to assessing the health of fuel cells[^thomas2014online]. I worked THD into some of my work using data-driven condition monitoring and will be looking at it more in the future, perhaps in another post.

# Classifying Mining Truck Oil Samples

<div style="text-align: center; ">
<figure>
  <img src="/assets/img/mining_dump_truck.jpg" alt="mining haul truck form India" style="background:none; border:none; box-shadow:none; text-align:center"/>
  <figcaption style="color:grey; font-size:smaller">Photo by <a href="https://www.pexels.com/@gunshe-ramchandani-992682?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels">Gunshe Ramchandani</a></figcaption>
</figure>
</div>

Maintenance on mining haul trucks is an expensive affair. But sudden failures are even more expensive -- costs can easily exceed $100,000 per failure. As such, regular oil samples are taken to monitor the health of these big trucks -- think of it like a occasional blood sample you get taken from the doctor. Once the oil samples are taken, though, someone needs to review the results to identify if there are significant concerns. In the research presented [here](https://doi.org/10.1016/j.ymssp.2014.12.020), the authors discuss how classifying the results, by hand, is time consuming. A model would go a long way in speeding up the review.[^phillips2015classifying]

The researchers used 400,000 hand-labelled oil sample results from large haul-truck diesel engines in Australia. They then selected six of the best variables from the results, variables like the amount of iron in the sample, to use in the model training. The authors trained both a logistic regression model and a support vector machine (SVM) model to either classify the oil sample from a healthy, or unhealthy, engine.

The logistic regression model outperformed the SVM model. The authors also make an interesting point about the explanatory power of the simpler logistic regression model, which I also find compelling:

"...the logistic regression model was simple to explain, there were no difficulties in explaining the method, and they could reconcile the weights of the explanatory variables with their understanding of failure modes for these engines." 

I like this example for its simplicity. No complicated feature engineering is needed. Rather, experts are asked to select the variable -- the features -- that are most indicative of engine health. No complicated model is used. Rather, a straightforward logistic regression is implemented. And finally, the authors are cognizant of how the results will be perceived by the customer. Trust is gained, between them and the end-user, by using model that can be interpreted.


# References

[^bizub2017methods]: Bizub, Jeffrey Jacob. "[Methods and systems to derive engine component health using total harmonic distortion in a knock sensor signal](https://patents.google.com/patent/US9791343B2/en)." U. S. Patent No. 9, 791, 343. 17 Oct. 2017.

[^f1993ieee]: F II, I. "[IEEE recommended practices and requirements for harmonic control in electrical power systems](http://www.coe.ufrj.br/~richard/Acionamentos/IEEE519.pdf)." New York, NY, USA (1993): 1-1.

[^thomas2014online]: Thomas, Sobi, et al. "[Online health monitoring of a fuel cell using total harmonic distortion analysis](https://doi.org/10.1016/j.ijhydene.2013.12.180)." international journal of hydrogen energy 39.9 (2014): 4558-4565.


[^phillips2015classifying]: Phillips, J., et al. "[Classifying machinery condition using oil samples and binary logistic regression](https://doi.org/10.1016/j.ymssp.2014.12.020)." Mechanical Systems and Signal Processing 60 (2015): 316-325.
