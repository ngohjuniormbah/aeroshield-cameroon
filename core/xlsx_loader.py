from __future__ import annotations

import datetime as dt
import io
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

NS_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

NUMERIC_COLUMNS = {
    "id",
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "apparent_temperature_mean",
    "daylight_duration",
    "sunshine_duration",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
    "latitude",
    "longitude",
}

DATETIME_COLUMNS = {"time", "sunrise", "sunset"}
TEXT_COLUMNS = {"city", "region"}


def excel_serial_to_datetime(serial: float) -> dt.datetime:
    """Convert Excel serial number to datetime using the 1900 date system."""
    return dt.datetime(1899, 12, 30) + dt.timedelta(days=float(serial))



def recover_day_month_decimal(serial_like: float) -> float:
    """
    Recover values that were accidentally converted by Excel from strings like 21.9 -> 21 Sep.
    Example: 46286 -> 21.09
    """
    recovered = excel_serial_to_datetime(serial_like)
    return float(f"{recovered.day}.{recovered.month:02d}")



def _load_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for si in root.findall(f"{NS_MAIN}si"):
        text = "".join((t.text or "") for t in si.iterfind(f".//{NS_MAIN}t"))
        strings.append(text)
    return strings



def load_hackathon_xlsx(path: str | Path) -> pd.DataFrame:
    """
    Load the hackathon Excel workbook robustly.

    Why this custom reader exists:
    some numeric values inside the provided workbook were auto-converted by Excel into date serials.
    This loader reads the raw XML and reconstructs those values safely.
    """
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        shared_strings = _load_shared_strings(zf)
        worksheet_xml = zf.read("xl/worksheets/sheet1.xml")

    context = ET.iterparse(io.BytesIO(worksheet_xml), events=("end",))
    letters = [chr(ord("A") + i) for i in range(26)]

    header: list[str] | None = None
    records: list[dict[str, object]] = []

    for _, elem in context:
        if elem.tag != f"{NS_MAIN}row":
            continue

        cells: dict[str, str | None] = {}
        for cell in elem.findall(f"{NS_MAIN}c"):
            ref = cell.get("r")
            if not ref:
                continue
            col = re.match(r"([A-Z]+)", ref).group(1)
            cell_type = cell.get("t")
            value_node = cell.find(f"{NS_MAIN}v")
            raw_value = value_node.text if value_node is not None else None
            if cell_type == "s" and raw_value is not None:
                raw_value = shared_strings[int(raw_value)]
            cells[col] = raw_value

        if header is None:
            header = [cells.get(col) for col in letters if cells.get(col) is not None]
            elem.clear()
            continue

        row_dict: dict[str, object] = {}
        for idx, column_name in enumerate(header):
            col_letter = letters[idx]
            raw = cells.get(col_letter)

            if column_name in NUMERIC_COLUMNS:
                if raw in (None, ""):
                    row_dict[column_name] = np.nan
                else:
                    numeric_value = float(raw)
                    if numeric_value > 1000 and column_name not in {"daylight_duration", "sunshine_duration"}:
                        row_dict[column_name] = recover_day_month_decimal(numeric_value)
                    else:
                        row_dict[column_name] = numeric_value
            elif column_name in DATETIME_COLUMNS:
                if raw in (None, ""):
                    row_dict[column_name] = pd.NaT
                else:
                    row_dict[column_name] = excel_serial_to_datetime(float(raw))
            elif column_name in TEXT_COLUMNS:
                row_dict[column_name] = raw
            else:
                row_dict[column_name] = raw

        records.append(row_dict)
        elem.clear()

    df = pd.DataFrame(records)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for time_col in ["sunrise", "sunset"]:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    return df
