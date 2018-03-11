# Comprex

[![CircleCI](https://circleci.com/gh/HamedMP/CompreX.svg?style=svg)](https://circleci.com/gh/HamedMP/CompreX)

A parameter-free anomaly detection using pattern-based compression
---

Anomaly detection as one of the tasks in unsupervised learning is a hard task by itself, and adding categorical data which most of the algorithms requires encoding them makes it harder. 

When dealing with high dimensional categorical data, features which have +1 million different values (which is my case) encoding becomes impractical. While researching for best approaches for dealing with this kind of datasets, I found **CompreX** [[1](1)] approach the most intuitive way to deal with the data by encode them using *shannon entropy* and *MDL (Minimum Description Length)*. 

[1]: ./resources/fast-anomaly.pdf	"L. Akoglu, H. Tong, J. Vreeken, and C. Faloutsos. Fast and reliable anomaly detection incategorical data. 2012."

## Dependencies

1. Numpy
2. Pandas
3. Scikit-learn

## Roadmap

- [x] Initial implementation, v1.0.0
- [ ] Check correctness! 
- [ ] Add `Docstrings`
- [ ] Publish documentation 
- [ ] Finish Scikit-learn style api compatibility issues
- [ ] Upload to PyPi

## Reference

[1] L. Akoglu, H. Tong, J. Vreeken, and C. Faloutsos. Fast and reliable anomaly detection incategorical data. 2012. ([read here](./resources/fast-anomaly.pdf))