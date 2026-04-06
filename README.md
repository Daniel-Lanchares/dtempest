# dtempest

The dtempest[¹] library is meant to provide a flexible implementation of an NPE[²] approach to parameter estimation 
through the use of normalizing flows.
It allows the user to train machine learning models to perform regression tasks on an arbitrary 
collections of images.
With the GW[³] package, this core functionality is applied to CBC[⁴] signals based on its q-transform.
Due to dependence on LALSuite algorithms the GW package is restricted to Linux/macOS operating systems.

[1]: **D**eep **T**ransform **E**xchangeable **M**odels for **P**osterior **Est**imation

[2]: **N**eural **P**osterior **E**stimation

[3]: **G**ravitational **W**aves

[4]: **C**ompact **B**inary **C**oalescence

This library has been developed as a Master of Science project at the University of Oviedo, available [here](https://digibuo.uniovi.es/dspace/handle/10651/74109), and whose later results are published in [Lanchares et al. 2025](https://arxiv.org/abs/2505.08089).

## Usage
The process is meant to have three stages:

- **Dataset generation**: Creation of a labeled dataset of CBC waveforms injected in real noise. 
- **Model training**: Normalizing flows train in much the same way as neural networks do, through gradient descent. The flow itself is highly configurable.
- **Model Inference** Once trained these models can perform regression tasks on either new datasets or, 
  more interestingly, real data. For gravitational waves, data is taken automatically from open data sources.

For a hands-on overview of the library and the helper scripts needed to make the plots in [Lanchares et al. 2025](https://arxiv.org/abs/2505.08089), see the examples.

## Main requirements
This code is built on **PyTorch** and relies on **glasflow.nflows** for its implementation of normalizing flows.

Gravitational wave utilities require various gw related libraries, so one of the **igwn conda 
environments** is advised as an installation starting point.