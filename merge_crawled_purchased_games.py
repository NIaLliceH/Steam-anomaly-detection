import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# 1. Đọc dữ liệu cũ và mới
log.info("Đọc file raw cũ...")
old_df = pd.read_csv("data/raw/purchased_games.csv")

log.info("Đọc file vừa crawl...")
# new_df = pd.read_csv("data/crawled/targeted_purchased_games.csv")
new_df = pd.read_csv("data/crawled/model_purchased_games.csv")

# 2. Nối (Append) 2 bảng lại với nhau
combined_df = pd.concat([old_df, new_df], ignore_index=True)

# 3. Chủ động xóa trùng lặp ngay tại đây (ưu tiên data mới nằm dưới cùng)
combined_df = combined_df.drop_duplicates(subset=["playerid"], keep="last")

# 4. Ghi đè lại vào file raw
combined_df.to_csv("data/raw/purchased_games.csv", index=False)
log.info(f"Hoàn tất! File raw mới có tổng cộng {len(combined_df)} players.")