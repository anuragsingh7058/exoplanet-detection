# Exoplanet Transit Detection Pipeline

Ever wondered how scientists find planets orbiting stars millions of light years away? They look for tiny dips in a star's brightness — when a planet passes in front of it, it blocks a little light. That's a transit.

The problem is the data is incredibly noisy. Real transit signals look almost identical to random instrument glitches, background star interference, or the star's own surface activity. Finding a planet in that noise is like spotting a specific raindrop in a thunderstorm.

This project builds an AI pipeline that does exactly that — automatically.

## What it does
- Loads raw brightness data (light curves) from NASA's Kepler telescope
- Cleans and normalizes the noisy signal
- Applies Gaussian smoothing to reduce sharp random spikes
- Uses SMOTE to fix the massive class imbalance (only 37 confirmed planets out of 2259 stars)
- Trains a 3-layer 1D CNN to classify each star as planet or no planet
- Achieves 100% planet recall by tuning the classification threshold to 0.3

## Results
99% overall accuracy. More importantly — 100% planet recall. We don't miss planets.

## Tech used
Python, TensorFlow/Keras, NumPy, Pandas, Matplotlib, SciPy, imbalanced-learn, Google Colab

## Dataset
NASA Kepler Labeled Time Series — publicly available on Kaggle

## Built for
ISRO Bharatiya Antariksh Hackathon (BAH) 2026
