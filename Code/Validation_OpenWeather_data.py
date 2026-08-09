# This file is part of the article:
# "Leveraging Remote Traffic Data for Local Air Pollutant Estimation:
# A Scenario-Based Machine Learning Study Across London Monitoring Sites"
#
# Copyright (C) 2026 The authors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.



import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr


def parse_defra_datetime(df, date_col="Date", time_col="Time"):
    """
    Parse DEFRA date and time columns into a single datetime column.

    Supported date formats:
        - mm-dd-yy
        - mm/dd/yy
        
    The value 24:00:00 is converted to 00:00:00 of the following day.

    Returns
    -------
    pandas.Series
        Datetime values (timezone-naive).
    """

    def parse_date(date_str):
        if pd.isna(date_str):
            return pd.NaT

        date_str = str(date_str).strip()

        if "-" in date_str:
            return pd.to_datetime(
                date_str,
                format="%m-%d-%y",
                errors="coerce"
            )

        # mm/dd/yy or mm/dd/yyyy
        if "/" in date_str:
            try:
                return pd.to_datetime(
                    date_str,
                    format="%m/%d/%y",
                    errors="raise"
                )
            except ValueError:
                return pd.to_datetime(
                    date_str,
                    format="%m/%d/%Y",
                    errors="coerce"
                )

        return pd.NaT

    dates = df[date_col].apply(parse_date)

    if dates.isna().any():
        raise ValueError(
            f"Invalid dates found: "
            f"{df.loc[dates.isna(), date_col].unique().tolist()}"
        )

    time_parts = (
        df[time_col]
        .astype(str)
        .str.strip()
        .str.split(":", expand=True)
    )

    if time_parts.shape[1] != 3:
        raise ValueError(
            "Time must have HH:MM:SS format."
        )

    hours = pd.to_numeric(time_parts[0], errors="coerce")
    minutes = pd.to_numeric(time_parts[1], errors="coerce")
    seconds = pd.to_numeric(time_parts[2], errors="coerce")


    datetime = (
        dates
        + pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(minutes, unit="m")
        + pd.to_timedelta(seconds, unit="s")
    )

    return datetime




dir_drive = r"UK-air defra data"
dir_weather = r"Bkg_or_with_traffic\Open_weather"
filepath2save = Path("Figs")
filepath2save.mkdir(exist_ok=True)

files = ["Camden Kerbside 23jun2025-12dec2025.csv", "London Marylebone Road 23jun2025-12dec2025.csv"]

station_labels = {"CamdenKerbside": "Camden Kerbside", "LondonMaryleboneRoad": "Marylebone Road"}

fig, axes = plt.subplots(2, 2, figsize=(18, 16))

for row_idx, file in enumerate(files):

    match = re.match(r"^[A-Za-z\s-]+", file)
    station = match.group().replace(" ", "")
    station_label = station_labels.get(station, station)

    df = pd.read_csv(os.path.join(dir_drive, file))
    df["datetime"] = parse_defra_datetime(df)

    keys = [c for c in df.columns if not c.startswith("Status") and c != "Station"]
    df = df[keys]
    df = df.replace(r"(?i)^no data$", np.nan, regex=True)

    cols = [df.columns[-1]] + list(df.columns[:-1])
    df = df[cols]

    df["datetime"] = pd.to_datetime(df["datetime"], errors="raise").dt.tz_localize("UTC")

    weather_file = "OpenWeather_" + station + ".csv"
    df_weather = pd.read_csv(os.path.join(dir_weather, weather_file))
    df_weather["datetime"] = pd.to_datetime(df_weather["date"], errors="raise", utc=True)

    df = df.merge(df_weather, on="datetime", how="inner")
    

    df["temp_ow"] = pd.to_numeric(df["temp_ow"], errors="coerce") - 273.15
    df["Modelled Temperature"] = pd.to_numeric(df["Modelled Temperature"], errors="coerce")
    df["wind_speed_ow"] = pd.to_numeric(df["wind_speed_ow"], errors="coerce")
    df["Modelled Wind Speed"] = pd.to_numeric(df["Modelled Wind Speed"], errors="coerce")

    # =========================================================
    # TEMPERATURE
    # =========================================================

    data = df[["Modelled Temperature", "temp_ow"]].dropna()

    x = data["Modelled Temperature"].to_numpy()
    y = data["temp_ow"].to_numpy()

    coef = np.polyfit(x, y, 1)
    r, _ = pearsonr(x, y)

    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    x_range = np.linspace(min_val, max_val, 200)

    ax = axes[row_idx, 0]

    ax.scatter(x, y, alpha=0.35, s=22)

    ax.plot(x_range, x_range, linestyle="--", linewidth=1.5, label="Perfect agreement")
    ax.plot(x_range, coef[0] * x_range + coef[1], linewidth=2.2, label=f"Linear fit (slope = {coef[0]:.2f})")

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    ax.set_xlabel(f"Temperature (°C) measured at the air quality monitoring station", fontsize=15)
    ax.set_ylabel("Temperature (°C) acquired from OpenWeather", fontsize=15)

    ax.set_title(f"{station_label} — Temperature", fontsize=18, pad=12)

    ax.text(0.05, 0.93, f"$r$ = {r:.2f}\n$n$ = {len(data)}", transform=ax.transAxes, fontsize=14, verticalalignment="top", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    ax.tick_params(axis="both", labelsize=13)
    ax.legend(fontsize=12, frameon=True)
    ax.grid(alpha=0.2)

    # =========================================================
    # WIND SPEED
    # =========================================================

    data = df[["Modelled Wind Speed", "wind_speed_ow"]].dropna()

    # Correlation matrix (Pearson by default)
    corr = data.corr(numeric_only=True)
    print(corr)

    x = data["Modelled Wind Speed"].to_numpy()
    y = data["wind_speed_ow"].to_numpy()

    coef = np.polyfit(x, y, 1)
    r, _ = pearsonr(x, y)

    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    x_range = np.linspace(min_val, max_val, 200)

    ax = axes[row_idx, 1]

    ax.scatter(x, y, alpha=0.35, s=22)

    ax.plot(x_range, x_range, linestyle="--", linewidth=1.5, label="Perfect agreement")
    ax.plot(x_range, coef[0] * x_range + coef[1], linewidth=2.2, label=f"Linear fit (slope = {coef[0]:.2f})")

    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)

    ax.set_xlabel("Wind speed (m/s) measured at the air quality monitoring station", fontsize=15)
    ax.set_ylabel("Wind speed (m/s) acquired from OpenWeather", fontsize=15)

    ax.set_title(f"{station_label} — Wind speed", fontsize=18, pad=12)

    ax.text(0.05, 0.93, f"$r$ = {r:.2f}\n$n$ = {len(data)}", transform=ax.transAxes, fontsize=14, verticalalignment="top", bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    ax.tick_params(axis="both", labelsize=13)
    ax.legend(fontsize=12, frameon=True)
    ax.grid(alpha=0.2)
plt.tight_layout()
fig.subplots_adjust(hspace=0.32)

fig.savefig(filepath2save / "OpenWeather_validation_temperature_wind.png", dpi=600, bbox_inches="tight", facecolor="white")

plt.show()
plt.close(fig)