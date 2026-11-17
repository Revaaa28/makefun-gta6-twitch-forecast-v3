# makefun-gta6-twitch-forecast-v3

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Revaaa28/makefun-gta6-twitch-forecast-v3/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Revaaa28/makefun-gta6-twitch-forecast-v3/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A launch-day Twitch viewer forecaster for Grand Theft Auto VI. Given everything recorded up to the end of hour 1 of release day, it forecasts five quantiles (5%, 25%, 50%, 75%, 95%) of the Twitch category's viewer count for each of hours 2 to 24.

## How it works

This entry is an ensemble of the two approaches that each worked on only half of the practice launches alone:

1. **Pool selection by unlock type.** The case file carries dozens of past big game launches with their full hourly curves. The program keeps only the launches whose unlock was a *rolling midnight local* (console launches like GTA VI, no simultaneous global unlock, no target-history pseudo-launch): those share GTA VI's day shape, while simultaneous-global launches follow a different curve and poison a pooled shape.
2. **Size-weighted local-linear estimation.** For every forecast hour it fits a Gaussian-kernel-weighted line of the log ratio log(viewers_h / viewers_1) against log10(viewers_1) over that pool, so a launch of GTA VI's size is forecast mostly from similar-sized launches. The quantiles are the weighted quantiles of the line's residuals, smoothed across neighbouring hours and widened by 5%. If the rolling pool ever came back too small, the pool widens to already-launched comparables, then to all complete ones.

## Validation

Graded locally on the kit's four practice launches against the real answers, using the exact contest pinball formula (log scale, levels 0.05/0.25/0.5/0.75/0.95):

| model | p00 | p01 | p02 | p03 | mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| **this ensemble** | 0.2302 | 0.1538 | 0.1603 | 0.2457 | **0.1975** |
| regression on launched pool | 0.4646 | 0.2470 | 0.1370 | 0.1888 | 0.2593 |
| median shape on rolling pool | 0.4013 | 0.1540 | 0.2080 | 0.2352 | 0.2496 |
| house persistence baseline | 0.7437 | 0.2868 | 0.2146 | 0.3033 | 0.3871 |

The ensemble's worst practice case (0.2457) is better than either parent model's worst case (0.40+): the combination is not just a better average, it removes each parent's catastrophic case. Deterministic, Python standard library only, no network.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=revaaa28&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
