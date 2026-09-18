"""Shared configuration for the pipeline."""

CITIES = [
    {"name": "Copenhagen", "latitude": 55.6761, "longitude": 12.5683},
    {"name": "London", "latitude": 51.5074, "longitude": -0.1278},
    {"name": "New York", "latitude": 40.7128, "longitude": -74.0060},
]

DB_PATH = "data/weather.duckdb"

# Open-Meteo is free and requires no API key: https://open-meteo.com/en/docs
API_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_FIELDS = "temperature_2m,precipitation,wind_speed_10m,relative_humidity_2m"
PAST_DAYS = 2
FORECAST_DAYS = 1
