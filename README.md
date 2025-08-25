# Aiyagari–Bewley–Huggett–Imrohoglu (ABHI) Model Analysis  

## Overview  
This repository implements and analyzes the **Aiyagari–Bewley–Huggett–Imrohoglu (ABHI) model**, a foundational framework in computational macroeconomics for studying household consumption–savings decisions under income uncertainty and borrowing constraints.  

The project integrates **theoretical derivations, Python-based simulation, and visualization**, enabling researchers to validate equilibrium dynamics, explore life-cycle behavior, and study the impact of borrowing constraints.  

---

## Mathematical Model  

Households maximize lifetime utility over a finite horizon \(T\):  

$$
\max_{\{c_t, a_{t+1}\}} \; \mathbb{E}\left[ \sum_{t=1}^T \beta^t \ln(c_t) \right]
$$

subject to the budget constraint:  

$$
c_t + a_{t+1} = (1+r)a_t + y_t, \quad a_t \geq a_{\min}
$$ 

with stochastic log-income dynamics:  

$$
\ln(y_{t+1}) = \rho \ln(y_t) + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2)
$$  

### Default Parameters  
- Interest rate $(r = 0.05)$  
- Discount factor $(\beta = 0.95)$  
- Time horizon $(T = 40)$  
- Minimum asset levels: $(a \in \{-40, -10, 0\})$  
- Income process: $(\rho = 0.90, \sigma = 0.1)$  
- Initial conditions: $(y_1 = 1, a_1 = 0)$, terminal condition $(a_{T+1} = 0)$  

---

## Repository Structure  

```
ABHI-Model-Analysis/
│── Documentation.pdf        # Full project report with derivations and explanations
│── Figure_1.png             # Plots: mean and variance of income, assets, consumption
│── Figure_2.png             # Plots: life-cycle distribution evolution
│── Figure_3.png             # Plots: asset/consumption under borrowing constraints
│── Life time plots.py       # Simulation of life-cycle distributions via Monte Carlo
│── Mean_Variance_Plots.py   # Baseline model solution: mean/variance plots
│── case_analysis.py         # Comparative analysis for different asset constraints
│── README.md                # Project overview (this file)
```
## Features  
- **Baseline solution** of the ABHI model with consumption smoothing and Euler equation dynamics.  
- **Monte Carlo life-cycle simulations** (100k+ households) with log-normal income shocks.  
- **Comparative statics** for borrowing limits ($$(a = -40, -10, 0)$$).  
- **Figures & visualizations** of income, consumption, and asset distributions.  
- **Performance benchmarking** for runtime efficiency.  

---

## Installation  

Clone this repository and install the required dependencies:  

```bash
git clone https://github.com/mananroychoudhury/ABHI-Model-Analysis.git
cd ABHI-Model-Analysis
pip install -r requirements.txt
```
## Results  

- **Income** converges to a log-normal distribution with long-run variance ≈ 0.0526.  
- **Consumption** follows a declining path, consistent with the Euler equation: \(c_{t+1} \approx 0.9975 c_t\).  
- **Assets** diverge over time with rising variance, reflecting heterogeneous household outcomes.  
- **Borrowing constraints** significantly alter both consumption trajectories and final wealth.  

---

## Performance  

- **Baseline model runtime:** ~0.017 sec  
- **Monte Carlo (100k households):** ~16 sec  
- **Borrowing constraint analysis:** ~0.15 sec  

Future scalability:  
- Vectorization of array operations.  
- Parallelization with `joblib` or `dask`.  
- Distributed simulations using **Ray** or **Apache Spark**.  

---

## Extensions and Future Work  

- Incorporating heterogeneity in interest rates, taxes, and inflation.  
- Introducing stochastic shocks to both income and expenditures.  
- Calibrating the model against real household panel datasets.  
- Scaling simulations to millions of agents using cloud and distributed computing.  

---

## References  

- Aiyagari, S.R. (1994). *Uninsured idiosyncratic risk and aggregate saving*.  
- Bewley, T. (1977). *The permanent income hypothesis: A theoretical formulation*.  
- Huggett, M. (1993). *The risk-free rate in heterogeneous-agent incomplete-insurance economies*.  

---

## License  

This project is licensed under the **MIT License**.  
