import argparse
import os
from argparse import Namespace

args: Namespace

def parse_args() -> Namespace:
    global  args
    p = argparse.ArgumentParser(description="Custom forecaster (null‑safe)")
    # Mongo
    p.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://admin:2059$tephan5203@94.130.171.244:27017/?authSource=admin&readPreference=secondaryPreferred&directConnection=true"))
    p.add_argument("--db", default=os.getenv("MONGO_DB", "PriceForecast"))
    p.add_argument("--coll-stock", default=os.getenv("COLL_STOCK", "DailyStockData"))
    p.add_argument("--coll-crypto", default=os.getenv("COLL_CRYPTO", "DailyCryptoData"))
    p.add_argument("--coll-fx", default=os.getenv("COLL_FX", "DailyCurrencyExchangeRates"))
    p.add_argument("--coll-news", default=os.getenv("COLL_NEWS", "NewsHeadlines"))
    p.add_argument("--coll-weather", default=os.getenv("COLL_WEATHER", "DailyWeatherData"))
    p.add_argument("--coll-pred", default=os.getenv("COLL_PRED", "PricePredictions"))
    # Horizons
    p.add_argument("--use-trading-days", action="store_true", default=os.getenv("USE_TRADING_DAYS", "false").lower()=="true")
    p.add_argument("--horizons-cal", type=str, default=os.getenv("HORIZONS_CAL", "7,30,182,365"))
    p.add_argument("--horizons-td", type=str, default=os.getenv("HORIZONS_TD", "5,21,126,252"))
    # Weather
    p.add_argument("--weather-pca", type=int, default=int(os.getenv("WEATHER_PCA", 12)))
    p.add_argument("--weather-agg", type=str, choices=["mean","median"], default=os.getenv("WEATHER_AGG", "mean"))
    p.add_argument("--fold-aware-weather-pca", action="store_true", default=os.getenv("FOLD_AWARE_WEATHER_PCA", "false").lower()=="true")
    # Model/training
    p.add_argument("--lookback", type=int, default=int(os.getenv("LOOKBACK", 90)))
    p.add_argument("--batch", type=int, default=int(os.getenv("BATCH", 128)))
    p.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", 15)))
    p.add_argument("--lr", type=float, default=float(os.getenv("LR", 7e-4)))
    p.add_argument("--dropout", type=float, default=float(os.getenv("DROPOUT", 0.15)))
    p.add_argument("--layers", type=int, default=int(os.getenv("LAYERS", 2)))
    p.add_argument("--hidden", type=int, default=int(os.getenv("HIDDEN", 128)))
    p.add_argument("--weight-decay", type=float, default=float(os.getenv("WEIGHT_DECAY", 1e-4)))
    p.add_argument("--seed", type=int, default=int(os.getenv("SEED", 42)))
    p.add_argument("--num-workers", type=int, default=int(os.getenv("NUM_WORKERS", 1)))
    # Split strategy
    p.add_argument("--walk-forward", action="store_true", default=os.getenv("WALK_FORWARD", "false").lower()=="true")
    p.add_argument("--train-span-days", type=int, default=int(os.getenv("TRAIN_SPAN_DAYS", 182)))
    p.add_argument("--val-span-days", type=int, default=int(os.getenv("VAL_SPAN_DAYS", 90)))
    p.add_argument("--step-days", type=int, default=int(os.getenv("STEP_DAYS", 90)))
    p.add_argument("--val-ratio", type=float, default=float(os.getenv("VAL_RATIO", 0.2)))
    # Quantiles
    p.add_argument("--use-quantiles", action="store_true", default=os.getenv("USE_QUANTILES", "false").lower()=="true")
    p.add_argument("--quantiles", type=str, default=os.getenv("QUANTILES", "0.5,0.9"))
    # News PCA cap
    p.add_argument("--news-pca-cap", type=int, default=int(os.getenv("NEWS_PCA_CAP", 32)))
    # Output / ranking
    p.add_argument("--top-k", type=int, default=int(os.getenv("TOP_K", 25)))
    p.add_argument("--rank-horizon", type=str, default=os.getenv("RANK_HORIZON", "1y"))
    p.add_argument("--rank-quantile", type=float, default=float(os.getenv("RANK_QUANTILE", 0.5)))
    p.add_argument("--ignore-days", type=int, default=int(os.getenv("IGNORE_DAYS", 0)))
    # Artifacts
    p.add_argument("--artifacts-dir", type=str, default=os.getenv("ARTIFACTS_DIR", "outputs"))

    args = p.parse_args()

    return args
