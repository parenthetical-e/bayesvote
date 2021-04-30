# bdcode
A general purpose bayesian decoding library, for time series.

# details
The underlying algorithm is based on ideas from a neuroscience paper,

> Insanally, M. N. et al. Spike-timing-dependent ensemble encoding by non-classically responsive cortical neurons. eLife 8, (2019).

We put these same ideaa to use, just now with a more general focus. The big idea is still very simple. In short to train we,

1. build a kernel probability dist, 
2. tune its bandwidth by CV, 
3. use the result to do a sequential bayesian decode. 

To test this scheme gets repeated, minus tuning, for rolling time windows.

That's it.

# usage 
TODO

# install
  git clone 
  pip install .
 
# dependencies
- scikit learn
- standard conda install 
