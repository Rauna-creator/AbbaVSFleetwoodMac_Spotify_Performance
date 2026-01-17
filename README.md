# ABBA vs Fleetwood Mac: Spotify Streaming Analysis

An academic data visualisation project analysing 336 songs from ABBA and Fleetwood Mac to understand streaming success drivers on Spotify.

## Overview

This project examines five decades of music (1970s-2020s) across 12 research questions about streaming performance, musical characteristics, and success factors.

## Key Findings

- ABBA demonstrates 49% higher popularity than Fleetwood Mac (43.8 vs 29.4, p < 0.001)
- Production quality (loudness) is strongest predictor of success (r = 0.320)
- Mamma Mia film-featured songs average 71.5 vs 40.2 for other ABBA tracks
- Musical mode shows no significant performance difference (p = 0.889)
- Mainstream pop-rock represents 60% of top 20 tracks
- Three success factors: loudness (+3.7 dB), energy (+0.113), danceability (+0.047)

## Dataset

- ABBA: 112 songs (1973-2021)
- Fleetwood Mac: 224 songs (1968-2003)
- Source: Spotify API audio features
- Features: Danceability, energy, valence, acousticness, loudness, tempo, key, mode

## Research Questions

1. Which artist dominates Spotify streaming?
2. How does musical era impact popularity?
3. Do these artists have distinct musical personalities?
4. How has song structure evolved over five decades?
5. Are happier songs more popular?
6. How have production characteristics evolved?
7. Which audio features predict popularity?
8. Do major vs minor keys influence performance?
9. What distinct song styles exist, and which perform best?
10. Does the Mamma Mia film franchise boost ABBA's streaming?
11. What separates hits from deep cuts?

## Methodology

- Statistical analysis: T-tests, correlation analysis, multiple regression
- Machine learning: K-means clustering (k=4) for style identification
- Visualisation: Comparative analysis, time series, distribution plots

## Installation

```bash
git clone https://github.com/yourusername/spotify-analysis.git
cd spotify-analysis
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
spotify-analysis/
├── data/
│   ├── Abba.csv
│   └── FleetwoodMac.csv
├── app.py
├── requirements.txt
└── README.md
```

## Dashboard Features

The Streamlit dashboard provides:
- Artist comparison with statistical tests
- Era-based performance analysis
- Musical personality profiles
- Feature importance analysis
- Cluster visualisation
- Mamma Mia effect analysis
- Hits vs deep cuts comparison

## Key Insights

### Production Over Content
Audio characteristics explain only 12.5% of popularity variance. Production quality (loudness) emerges as the strongest predictor, whilst emotional tone (valence) shows weak correlation (r = 0.093).

### Streaming Advantage
ABBA's advantage partially stems from cultural media exposure. The 13 Mamma Mia featured songs (mean: 71.5) substantially elevate ABBA's catalogue average. Fleetwood Mac lacks comparable mainstream film presence.

### Style Over Mode
Musical mode demonstrates no predictive power (p = 0.889), but musical style significantly influences success. High-energy dance and mainstream pop-rock over-represent in top performers.

### Success Factors
Analysis reveals three significant differentiators between hits and deep cuts:
1. Loudness: +3.7 dB (p < 0.001)
2. Energy: +0.113 (p = 0.002)
3. Danceability: +0.047 (p = 0.02)

## Limitations

- Spotify popularity reflects current streaming, not historical chart performance
- Causation cannot be inferred from correlation
- Sample sizes vary across eras (70s: 225 songs, 90s: 26 songs)
- Analysis limited to two artists from specific genres

## Academic Context

MSc Data Science coursework, Winter 2025/2026. Analysis employs rigorous statistical methods with data-driven insights.

## Licence

Academic use. Data from Spotify API subject to their terms of service.
