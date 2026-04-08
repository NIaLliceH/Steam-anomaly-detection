import csv
import json
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def read_purchased_csv(path: str) -> pd.DataFrame:
    """
    Read a purchased_games CSV that may contain malformed rows where the
    library JSON column has unescaped inner quotes (single " instead of "").

    Falls back to manual line-by-line repair for rows that don't parse cleanly.
    """
    csv.field_size_limit(10_000_000)

    rows = []
    bad_count = 0

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header

        for lineno, row in enumerate(reader, start=2):
            if len(row) == 2:
                rows.append({"playerid": row[0], "library": row[1]})
            else:
                bad_count += 1

    if bad_count:
        log.warning("%d malformed rows detected in %s — attempting repair…", bad_count, path)

        with open(path, "r", encoding="utf-8") as f:
            next(f)  # skip header
            for lineno, raw_line in enumerate(f, start=2):
                raw_line = raw_line.rstrip("\n")
                # Quick check: if it parses fine, it was already captured above
                try:
                    parsed = next(csv.reader([raw_line]))
                    if len(parsed) == 2:
                        continue  # already captured
                except Exception:
                    pass

                # Repair: split on first comma; strip outer quotes from library
                comma_idx = raw_line.index(",")
                playerid = raw_line[:comma_idx]
                rest = raw_line[comma_idx + 1:]

                if rest.startswith('"') and rest.endswith('"'):
                    library_str = rest[1:-1]
                else:
                    library_str = rest

                try:
                    json.loads(library_str)  # validate JSON
                    rows.append({"playerid": playerid, "library": library_str})
                except json.JSONDecodeError:
                    log.warning("  Line %d: could not repair (skipping). playerid=%s", lineno, playerid)

        log.info("Repair complete.")

    df = pd.DataFrame(rows)
    df["playerid"] = pd.to_numeric(df["playerid"])
    return df


# 1. Đọc dữ liệu cũ và mới
log.info("Đọc file raw cũ...")
old_df = pd.read_csv("data/raw/purchased_games.csv")

log.info("Đọc file vừa crawl...")
new_df = read_purchased_csv("data/crawled/targeted_purchased_games.csv")

# 2. Nối (Append) 2 bảng lại với nhau
combined_df = pd.concat([old_df, new_df], ignore_index=True)

# 3. Chủ động xóa trùng lặp ngay tại đây (ưu tiên data mới nằm dưới cùng)
combined_df = combined_df.drop_duplicates(subset=["playerid"], keep="last")

# 4. Ghi đè lại vào file raw
combined_df.to_csv("data/raw/purchased_games.csv", index=False)
log.info(f"Hoàn tất! File raw mới có tổng cộng {len(combined_df)} players.")
