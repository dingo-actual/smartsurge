# SmartSurge Documentation

Welcome to the SmartSurge documentation. SmartSurge is an enhanced requests library, utilizing Bayesian adaptive rate limit estimation.

## Overview

SmartSurge extends the functionality of the Python requests library with these key features:

- **Automatic Rate Limit Detection and Enforcement**: Automatically detects and respects rate limits.
- **Bayesian Rate Limit Estimation**: Uses Bayesian statistics for rate limit estimation.
- **Streaming Requests**: Support for resumable streaming requests.
- **Robust Error Handling**: Comprehensive error handling and logging.
- **Async Support**: Asynchronous request capabilities via aiohttp.

## Why SmartSurge?

When working with APIs that have rate limits but don't clearly document them, developers often have to implement ad-hoc rate limiting solutions. SmartSurge solves this problem by:

1. Automatically detecting rate limits through a principled search procedure
2. Using Bayesian statistics to estimate rate limits with confidence intervals
3. Adapting to changing rate limits in real-time
4. Providing a simple and ergonomic API that's compatible with the requests library

## Getting Started

Check out the [Installation](installation.md) and [Usage](usage.md) guides to get started with SmartSurge.
