# roblox-up-and-coming-forecast

Forecasts the Roblox Up-and-Coming chart's top 10 (computer, US) for the
72 hours after entries close — an entry for make.fun's
`roblox-up-and-coming-forecast-01` ($500).

`python3 main.py` reads `/input/history.ndjson` + `/input/task.json`
(or `$MAKEFUN_INPUT_DIR`) and writes `/output/forecast.json`
(or `$MAKEFUN_OUTPUT_DIR`): one per-place probability forecast + top10.
Python 3.13 standard library only, no network, fully deterministic.

Method: two rank forecasts blended 50/50 per place — (A) a Gaussian kernel
around a damped least-squares rank-velocity extrapolation (with a
playing-growth boost), uncertainty growing with horizon and volatility;
(B) an empirical binned rank-transition matrix learned from snapshot pairs
about one horizon apart. Each place keeps 0.30 mass unnamed so surprise
entrants still score above the floor.

Backtest on the dev history (log score, higher is better):
train<=Nov 4, test Nov 5-7: -41.0 vs persistence -55.8;
train<=Nov 2, test Nov 3-7: -43.7 vs persistence -55.7.
