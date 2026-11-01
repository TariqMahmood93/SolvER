# SolvIER: **Solv**ing **I**ncomplete **E**ntity **R**esolution

SolvER is an algorithm that combines transformer-based data imputation with state-of-the-art entity resolution. By imputing missing values before matching, SolvIER consistently improves F1-score over standard ER baselines — with up to **+17.4% relative improvement** on real-world benchmarks at high missingness rates.

## Overview

Entity Resolution (ER) — the task of identifying records that refer to the same real-world entity — is significantly challenged by missing attribute values. SolvER addresses this by integrating an imputation model as a pre-processing step before running [Ditto](https://github.com/megagonlabs/ditto).


## Requirements

- Python 3.8+
- PyTorch 1.9+
- HuggingFace Transformers
- Spacy with the `en_core_web_lg` model
- NVIDIA Apex (for mixed-precision training)

```bash
git clone https://github.com/TariqMahmood93/SolvIER.git
cd solvIER
pip install -r requirements.txt
```

## Results

SolvER is evaluated on six standard ER benchmarks under MCAR missingness rates of 5%, 10%, 20%, and 40%.

| Dataset         | 5% nulls | 10% nulls | 20% nulls | 40% nulls |
|-----------------|----------|-----------|-----------|-----------|
| DBLP-ACM        | +0.5%    | +1.4%     | +1.4%     | +0.8%     |
| DBLP-Scholar    | +0.1%    | +0.6%     | +1.2%     | +1.2%     |
| Fodors-Zagats   | +0.3%    | +5.2%     | +2.4%     | +17.4%    |
| Walmart-Amazon  | +1.0%    | +0.2%     | +1.3%     | +5.9%     |
| iTunes-Amazon   | +0.5%    | +0.8%     | +1.0%     | 1.0%      |
| BeerAdvo-RateBeer | +1.2%  | +1.7%     | +2.3%     | +1.3%     |

*Relative F1-score improvement of SolvIER over Ditto baseline.*
