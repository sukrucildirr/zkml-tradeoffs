import polars as pl
from giza_datasets import DatasetsLoader
import itertools

def load_data():
    token_data = DatasetsLoader().load('tokens-daily-prices-mcap-volume').filter(pl.col("token") == "WETH")
    return token_data

def extract_features(token_data):
    LAG = 1
    DAYS = [1, 3, 7, 14, 30]
    token_price_trend = token_data.with_columns(
        ((pl.col("price").shift(-LAG) - pl.col("price")) > 0).cast(pl.Int8).alias("target")
    ).with_columns(
        list(itertools.chain(*[
            (
                (pl.col("price").diff(n=days).alias(f"price_diff_{days}_days")),
                ((pl.col("price") - pl.col("price").shift(days)) > 0).cast(pl.Int8).alias(f"trend_{days}_days")
            ) for days in DAYS
        ]))
    ).with_columns([
        pl.col("date").dt.weekday().alias("day"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.year().alias("year")
    ])
    return token_price_trend

def main():
    token_data = load_data()
    features = extract_features(token_data)
    features.write_csv("data/token_features.csv")

if __name__ == "__main__":
    main()