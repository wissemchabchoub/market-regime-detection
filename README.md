<h3 align="center">Market Regime Detection</h3>

---

[Wissem Chabchoub](https://www.linkedin.com/in/wissem-chabchoub/) | [Contact us](mailto:chb.wissem@gmail.com)

## 📝 Table of Contents

- [📝 Table of Contents](#-table-of-contents)
- [🧐 About <a name = "about"></a>](#about)
- [🎥 Repository Structure  <a name = "repo-struct"></a>](#repo-struct)


## 🧐 About <a name = "about"></a>

In this project, we implement the [WK-means market regime clustering algorithm](https://arxiv.org/abs/2110.11848?context=q-fin). In addition, we compare the performance of this algorithm to known market regime clustering algorithms such as Moments k-Means and HMM.
The models are tested on generated intraday data.
Finally, we test the WK-means on the SP500 index.

<p align="center">
  <img src="data/spy.png?raw=true" />
    <em>SP500 market regimes</em>
</p>

<p align="center">
  <img src="data/mean_vol.png?raw=true" />
    <em>SP500 regimes mean-vol plot</em>
</p>


<p align="center">
  <img src="data/W_p.png?raw=true" />
</p>

<p align="center">
   <em>Wasserstein Distance</em>
</p>



## 🎥 Repository Structure  <a name = "repo-struct"></a>

1. `market_regime_detection.ipynb`: A Jupyter Notebook to test HMM vs. WK-means vs. Moments k-means on generated data.
2. `WK-means_sp500.ipynb`: A Jupyter Notebook to test the WK-means on the SP500 data.
3. `requirements.txt`: Requirements file
3. `src`: Source code folder
3. `data`: Data folder
