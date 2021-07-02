---
title: "The Business Case - Advances in Condition Monitoring, Pt I"
date: "2020-09-29 10:15"
categories: [Condition Monitoring, Business Case]
tags: [manufacturing, ai, machine learning, condition monitoring]     # TAG names should always be lowercase
description: "Why care about data-driven condition monitoring techniques? In this post, we articulate the business case for using these advanced methods in an industrial environment."
image: "/assets/img/condition_monitoring_splash.svg"
---

<div style="text-align: center; ">
<figure>
  <img src="/assets/img/gears.png" alt="gear" style="background:none; border:none; box-shadow:none; text-align:center"/>
</figure>
</div>

> Why care about data-driven condition monitoring techniques? In this post, we articulate the business case for using these advanced methods in an industrial environment. This is the first article in a series on advances in condition monitoring.

A critical 900 horsepower motor suddenly failed resulting in millions of dollars in lost revenue, and it happened on my watch. In reality, the failure wasn't attributed to any one person -- it was a system failure. But the missed "signals" along the way, those signals indicated a severe problem, were perhaps the worst part of the failure. A gut punch. The temperature probe on the thrust bearing showed multiple large spikes weeks before the failure. However, we were not equipped -- I was not equipped -- to identify those spikes amidst the vast quantity of data that was being collected across thousands of pieces of equipment. It was akin to finding a needle in a haystack.  

You may have a similar story of equipment failure that has cost your business immense grief, both in terms of money and effort. [Condition monitoring](https://en.wikipedia.org/wiki/Condition_monitoring) is a method that measures the parameters of machines -- such as temperatures, vibrations, pressures, etc. -- in order to detect faults in the machinery. However, yesterday's implementation of condition monitoring, using only vibration routes and/or high-level alarms, is ill equipped to manage the deluge of data that is being produced in today's modern industrial environment. To find those valuable "needles" in a haystack one must use more advanced methods. 

<br>
<div style="text-align: center;">
<figure>
  <img src="/assets/img/condition_monitoring.svg" alt="condition monitoring" style="background:none; border:none; box-shadow:none; text-align:center"/>
  <figcaption style="color:grey; font-size:smaller"><br>Several parameters that can be used in condition monitoring. (image from author)</figcaption>
</figure>
</div>
<!-- <br> -->

<!-- <figure>
  <img src="/images/condition_monitoring.svg" alt="condition monitoring" style="background:none; border:none; box-shadow:none;" class="center"/>
  <figcaption>Different parameters used in condition monitoring.</figcaption>
</figure> -->

<!-- ![svg](/images/condition_monitoring.png "examples of parameters in condition monitoring") -->

This series will explore the advanced data-driven techniques that can make use of this data deluge. We will give an overview of the two predominant approaches in condition monitoring; that is, feature engineering, and end-to-end deep learning. As an illustration, we will investigate the problem of detecting worn tools in metal machining. We will approach this problem using end-to-end deep learning. All the code will be provided, so if you are so inclined, you can follow along too. 

A [McKinsey study](https://www.mckinsey.com/business-functions/operations/our-insights/manufacturing-analytics-unleashes-productivity-and-profitability) estimated that the appropriate use of data-driven techniques by process manufacturers "typically reduces machine downtime by 30 to 50 percent and increases machine life by 20 to 40 percent".  And believe it or not, an industrial environment is one of the best places to make use of these new "artificial intelligence" techniques (as [recently illustrated in the Economist](https://www.economist.com/technology-quarterly/2020/06/11/businesses-are-finding-ai-hard-to-adopt)). The sheer availability of data in industrial environments, and the clear match of theory to application, make the business case for advanced condition monitoring techniques compelling. It is here, at the intersection of traditional industry and machine learning, that we unlock incredible value.  

Buckle up! 

<!-- <br/><br/>

- - - -

<br/><br/>
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>. -->

