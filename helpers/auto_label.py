import pandas as pd
import os

# 1. Đọc lô 50 tài khoản nghi ngờ mới nhất
new_df = pd.read_csv('outputs/to_review.csv')
new_df['human_label'] = 1  # Gán nhãn Bot toàn bộ

# 2. Xử lý cộng dồn (Append) vào file cũ
reviewed_path = 'data/reviewed.csv'
if os.path.exists(reviewed_path):
    # Nếu đã có file từ vòng trước, đọc nó lên
    old_df = pd.read_csv(reviewed_path)
    
    # Nối data mới vào đáy data cũ
    combined_df = pd.concat([old_df, new_df], ignore_index=True)
    
    # Xóa các dòng bị trùng lặp ID (giữ lại kết quả đánh giá mới nhất)
    combined_df = combined_df.drop_duplicates(subset=['playerid'], keep='last')
    
    # Ghi đè lại file
    combined_df.to_csv(reviewed_path, index=False)
    print(f"[+] Đã gộp thành công! Tổng số tài khoản trong sổ tay AI: {len(combined_df)}")
else:
    # Nếu chạy lần đầu chưa có file, lưu mới luôn
    new_df.to_csv(reviewed_path, index=False)
    print(f"[+] Đã tạo sổ tay AI mới (reviewed.csv) với {len(new_df)} tài khoản.")