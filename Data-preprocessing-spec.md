# Data Preprocessing Specification

Tài liệu này mô tả toàn bộ logic tiền xử lí cho 4 bảng dữ liệu thô trước khi đưa vào
feature engineering. Mục đích: hỗ trợ tích hợp vào Pentaho (hoặc pipeline tự động hóa
tương đương) khi có data mới crawl về.

---

## Dependency chung: `private_steamids.csv`

| Column | Type | Mô tả |
|--------|------|-------|
| `playerid` | INT64 | Steam ID của player cần loại khỏi tất cả bảng |

**Áp dụng cho:** tất cả 4 bảng — filter trước khi export.

---

## 1. `history.csv`

### Input schema
| Column | Type raw | Ghi chú |
|--------|----------|---------|
| `playerid` | INT64 | Steam 64-bit ID |
| `achievementid` | STRING | Dạng `<gameid>_<name>` (vd: `403640_ACH_1`) |
| `date_acquired` | STRING | Format: `YYYY-MM-DD HH:MM:SS` |

### Transformations
1. **Parse datetime**: `date_acquired` → TIMESTAMP (`YYYY-MM-DD HH:MM:SS`)
2. **Extract gameid**: regex `^(\d+)_` từ `achievementid` → INT32 nullable
   - Rows không match regex → gameid = NULL (giữ lại, không drop)
3. **Filter private**: loại rows có `playerid` trong `private_steamids.csv`
4. **Dedup**: `DISTINCT ON (playerid, achievementid, date_acquired)` — keep last

### Output schema (`history.parquet`)
| Column | Type | Ghi chú |
|--------|------|---------|
| `playerid` | INT64 | |
| `achievementid` | STRING | |
| `date_acquired` | TIMESTAMP | |
| `gameid` | INT32 nullable | NULL nếu achievementid không có prefix số |

### Data quality notes
- Không có null trong input (validated EDA)
- Không có duplicate (playerid, achievementid, date_acquired) trong raw data
- Date range: 2008-09-13 → 2026-04-02 (hợp lệ, Steam ra mắt 2003)

---

## 2. `players.csv`

### Input schema
| Column | Type raw | Ghi chú |
|--------|----------|---------|
| `playerid` | INT64 | |
| `country` | STRING | Full country name (ISO 3166 official), không phải ISO-2 |
| `created` | STRING | Format: `YYYY-MM-DD HH:MM:SS` |

### Transformations
1. **Parse datetime**: `created` → TIMESTAMP (`YYYY-MM-DD HH:MM:SS`, errors → NULL)
2. **Filter private**: loại rows có `playerid` trong `private_steamids.csv`
   - Lưu ý: players.csv CÓ chứa private players (khác với history.csv) — filter là cần thiết
3. **Dedup**: `DISTINCT ON (playerid)` — keep last

### Output schema (`players.parquet`)
| Column | Type | Ghi chú |
|--------|------|---------|
| `playerid` | INT64 | |
| `country` | STRING | 25.3% NULL — downstream dùng UTC offset = 0 làm fallback |
| `created` | TIMESTAMP | 1.41% NULL → NaN cho account age features |

### Data quality notes
- **Country format**: full name (vd: "Russian Federation", "United States"), KHÔNG phải ISO-2
  - Một số entries dùng ISO 3166 official name: "Iran, Islamic Republic of", "Viet Nam"
  - 7 entries dùng ISO-2 code (legacy) — cần support cả hai
- 3 duplicate playerid trong raw → xử lý bằng DISTINCT ON

---

## 3. `reviews.csv`

### Input schema
| Column | Type raw | Ghi chú |
|--------|----------|---------|
| `reviewid` | INT32 | Primary key |
| `playerid` | INT64 | |
| `gameid` | INT32 | |
| `review` | STRING | Nội dung review, 0.52% NULL |
| `helpful` | INT32 | Số votes helpful |
| `funny` | INT32 | Số votes funny |
| `awards` | INT32 | Số awards |
| `posted` | STRING | Format: `YYYY-MM-DD` |

### Transformations
1. **Parse datetime**: `posted` → DATE (`YYYY-MM-DD`)
2. **Filter private**: loại rows có `playerid` trong `private_steamids.csv`
3. **Dedup**: `DISTINCT ON (reviewid)` — keep last
   - Lưu ý: 14 cặp `(playerid, gameid)` có 2 reviewid khác nhau — KHÔNG drop, là reviews hợp lệ

### Output schema (`reviews.parquet`)
| Column | Type | Ghi chú |
|--------|------|---------|
| `reviewid` | INT32 | |
| `playerid` | INT64 | |
| `gameid` | INT32 | |
| `review` | STRING | 0.52% NULL — downstream treat as empty string |
| `helpful` | INT32 | 65.8% = 0 |
| `funny` | INT32 | 89.0% = 0 |
| `awards` | INT32 | 92.2% = 0 |
| `posted` | DATE | |

### Data quality notes
- `helpful/funny/awards` đã phân tích — không dùng trong feature engineering (drop ở model layer)
- NULL review text → `review_length = 0`, `review_duplication_rate` tính trên normalized text

---

## 4. `purchased_games.csv`

### Input schema
| Column | Type raw | Ghi chú |
|--------|----------|---------|
| `playerid` | INT64 | |
| `library` | STRING | JSON array — xem format bên dưới |

### Library formats (mixed trong cùng file)

**Format mới (90.4%)** — có playtime:
```json
[{"appid": 10, "playtime_mins": 150}, {"appid": 20, "playtime_mins": 0}, ...]
```

**Format cũ (6.8%)** — chỉ có appid:
```json
[10, 20, 30, ...]
```
→ Cần normalize thành: `[{"appid": 10, "playtime_mins": -1}, ...]`
(`playtime_mins = -1` là sentinel: "unknown playtime", phân biệt với 0 = "never played")

**NULL/Empty (2.8%)**: `library = NULL` hoặc `[]` → parsed thành empty list

### Transformations
1. **Repair malformed rows**: một số rows có unescaped inner quotes trong JSON
   - Logic: tìm vị trí comma đầu tiên → split thành `playerid` + `library`
   - Strip outer quotes của library field nếu có
   - Validate JSON; rows không repair được → skip với warning
2. **Parse JSON library**:
   - Detect format (có `"appid"` key → new format, array of int → old format)
   - Normalize về `[{appid: INT, playtime_mins: INT}, ...]`
   - `playtime_mins = -1` cho old format (unknown)
3. **Compute library_size**: `COUNT(items in parsed array)`
4. **Filter private**: loại rows có `playerid` trong `private_steamids.csv`
5. **Dedup**: `DISTINCT ON (playerid)` — keep last

### Output schema (`purchased.parquet`)
| Column | Type | Ghi chú |
|--------|------|---------|
| `playerid` | INT64 | |
| `library` | ARRAY<STRUCT<appid INT, playtime_mins INT>> | Normalized |
| `library_size` | INT32 | 0 cho null/empty library |

### Data quality notes
- Old format players (6.8%): `playtime_mins = -1` → downstream set NaN cho playtime features
- `playtime_mins = 0`: "có trong library nhưng chưa bao giờ chơi" (SAM signal)
- `playtime_mins > 0`: "đã chơi, đơn vị phút"
- `playtime_mins = -1`: "không có thông tin playtime" (old format)

---

## Thứ tự thực thi trong Pentaho

```
private_steamids.csv
        │
        ├──► history.csv   → history.parquet
        ├──► players.csv   → players.parquet
        ├──► reviews.csv   → reviews.parquet
        └──► purchased_games.csv → purchased.parquet
```

Các bảng **độc lập** với nhau trong bước preprocessing — có thể chạy song song.
Không có join giữa các bảng ở bước này; join được thực hiện ở bước feature engineering.

---

## Các vấn đề cần lưu ý khi có data mới

| Vấn đề | Bảng | Xử lý |
|--------|------|-------|
| Tên quốc gia thay đổi (vd: "Turkey" → "Türkiye") | players | Cập nhật country offset lookup table |
| Library format thay đổi | purchased | Kiểm tra có key mới ngoài `appid`/`playtime_mins` |
| Malformed CSV rows tăng | purchased | Monitor bad_lines count trong repair step |
| Duplicate playerid tăng | players | Expected khi crawl incremental; DISTINCT ON xử lý |
| Private players mới | tất cả | Cần refresh `private_steamids.csv` trước khi chạy |
