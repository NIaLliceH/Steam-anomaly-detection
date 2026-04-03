import sys
import requests
import pandas as pd
from datetime import datetime
import time
import os
import csv
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re

# ==========================================
# CONFIGURATION
# ==========================================

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")
STEAM_IDS = [
    76561198287996067,
    76561199761358443,
    76561198399223263,
    76561198350357346
]
OUTPUT_DIR = "data/crawled"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def unix_to_datetime(unix_ts):
    if not unix_ts:
        return None
    return datetime.fromtimestamp(unix_ts).strftime('%Y-%m-%d %H:%M:%S')

def get_json(url, params):
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"[-] API Error: {e}")
        return None

# ==========================================
# CRAWLING FUNCTIONS
# ==========================================
def crawl_player_info(steam_id):
    print("[+] Crawling Player Summary...")
    url = "http://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/"
    data = get_json(url, {"key": API_KEY, "steamids": steam_id})
    
    if not data or 'response' not in data or not data['response']['players']:
        print("[-] Profile is private or invalid.")
        return None
    
    p = data['response']['players'][0]
    return pd.DataFrame([{
        "playerid": steam_id,
        "country": p.get("loccountrycode", "Unknown"),
        "created": unix_to_datetime(p.get("timecreated"))
    }])

def crawl_library(steam_id):
    print("[+] Crawling Purchased Games (Library)...")
    url = "http://api.steampowered.com/IPlayerService/GetOwnedGames/v0001/"
    params = {
        "key": API_KEY,
        "steamid": steam_id,
        "include_appinfo": 0,
        "include_played_free_games": 1
    }
    data = get_json(url, params)
    
    if not data or 'response' not in data or 'games' not in data['response']:
        return None, []
    
    app_ids = [game['appid'] for game in data['response']['games']]
    
    df = pd.DataFrame([{
        "playerid": steam_id,
        "library": str(app_ids)  # Format list thành chuỗi như dataset Kaggle
    }])
    return df, app_ids

def crawl_achievements(steam_id, app_ids):
    print(f"[+] Crawling Achievements for {len(app_ids)} games (This might take a while...)")
    url = "http://api.steampowered.com/ISteamUserStats/GetPlayerAchievements/v0001/"
    history_records = []
    
    for i, app_id in enumerate(app_ids):
        # In tiến độ để tránh sốt ruột
        if i % 10 == 0 and i > 0:
            print(f"    ... processed {i}/{len(app_ids)} games")
            
        params = {"key": API_KEY, "steamid": steam_id, "appid": app_id}
        data = get_json(url, params)
        
        # Ngủ 0.2s để tránh bị Steam khóa IP vì Rate Limit (HTTP 429 Too Many Requests)
        time.sleep(0.2)
        
        if not data or 'playerstats' not in data:
            continue
            
        stats = data['playerstats']
        if not stats.get('success', False) or 'achievements' not in stats:
            continue
            
        for ach in stats['achievements']:
            if ach['achieved'] == 1:  # Chỉ lấy những thành tựu đã mở khóa
                history_records.append({
                    "playerid": steam_id,
                    "achievementid": f"{app_id}_{ach['apiname']}", # Nối appid và apiname
                    "date_accquired": unix_to_datetime(ach['unlocktime'])
                })
                
    return pd.DataFrame(history_records)

def crawl_reviews(steam_id):
    print("[+] Crawling Reviews via Web Scraping...")
    reviews_data = []
    page = 1
    
    while True:
        # Bắt buộc ép ?l=english để chữ "Posted" và "helpful" hiển thị tiếng Anh, dễ cào
        url = f"https://steamcommunity.com/profiles/{steam_id}/recommended/?p={page}&l=english"
        
        try:
            # Thêm User-Agent để Steam không chặn
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                break
                
            soup = BeautifulSoup(response.text, 'html.parser')
            review_boxes = soup.find_all('div', class_='review_box')
            
            # Nếu trang không còn review nào -> thoát vòng lặp
            if not review_boxes:
                break
                
            for box in review_boxes:
                # 1. Game ID
                game_link = box.find('div', class_='leftcol').find('a')
                game_id = game_link['href'].split('/')[-1] if game_link else 0
                
                # 2. Nội dung review (ẩn đi tag spoiler nếu có)
                content_div = box.find('div', class_='content')
                review_text = content_div.get_text(separator=" ", strip=True) if content_div else ""
                
                # 3. Ngày post
                posted_div = box.find('div', class_='posted')
                posted_text = posted_div.get_text(strip=True) if posted_div else ""
                posted_match = re.search(r'Posted (.*?)\.', posted_text)
                posted_date = parse_steam_date(posted_match.group(1)) if posted_match else None
                
                # 4. Review ID (Nằm trong ID của nút Like)
                vote_btn = box.find('a', id=re.compile(r'RecommendationVoteUpBtn\d+'))
                review_id = vote_btn['id'].replace('RecommendationVoteUpBtn', '') if vote_btn else 0
                
                # 5. Lượt Helpful & Funny
                header_div = box.find('div', class_='header')
                header_text = header_div.get_text(strip=True) if header_div else ""
                
                helpful_match = re.search(r'([\d,]+) (person|people) found this review helpful', header_text)
                helpful_count = int(helpful_match.group(1).replace(',', '')) if helpful_match else 0
                
                funny_match = re.search(r'([\d,]+) (person|people) found this review funny', header_text)
                funny_count = int(funny_match.group(1).replace(',', '')) if funny_match else 0
                
                reviews_data.append({
                    "reviewid": review_id,
                    "playerid": steam_id,
                    "gameid": game_id,
                    "review": review_text,
                    "helpful": helpful_count,
                    "funny": funny_count,
                    "awards": 0, # Web bỏ ẩn số lượng award, để 0 cho an toàn Data Pipeline
                    "posted": posted_date
                })
            
            print(f"    ... scraped page {page} ({len(review_boxes)} reviews)")
            page += 1
            time.sleep(0.5) # Chờ 0.5s để không bị Steam Rate Limit
            
        except Exception as e:
            print(f"[-] Error scraping reviews on page {page}: {e}")
            break
            
    df = pd.DataFrame(reviews_data)
    # Đảm bảo output luôn có đủ cột dù không cào được gì
    if df.empty:
        df = pd.DataFrame(columns=["reviewid", "playerid", "gameid", "review", "helpful", "funny", "awards", "posted"])
    
    print(f"    -> Total reviews scraped: {len(df)}")
    return df
    
def is_already_crawled(steam_id):
    """Kiểm tra xem STEAM_ID đã tồn tại trong file players.csv chưa"""
    filepath = os.path.join(OUTPUT_DIR, "players.csv")
    if not os.path.exists(filepath):
        return False
    
    try:
        # Chỉ đọc đúng cột playerid để tiết kiệm RAM
        df = pd.read_csv(filepath, usecols=["playerid"], dtype={"playerid": str}, encoding='utf-8')
        return str(steam_id) in df["playerid"].values
    except Exception as e:
        print(f"[!] Warning checking existence: {e}")
        return False

def save_append(df, filename):
    """Hàm lưu nối tiếp vào CSV. Tự động tạo header nếu file chưa tồn tại."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        df.to_csv(filepath, index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
    else:
        df.to_csv(filepath, mode='a', header=False, index=False, encoding='utf-8', quoting=csv.QUOTE_NONNUMERIC)
        
        
def parse_steam_date(date_str):
    """Chuẩn hóa chuỗi ngày của Steam về định dạng YYYY-MM-DD"""
    if not date_str: return None
    try:
        if "," in date_str:
            # Định dạng có năm: "21 June, 2025"
            dt = datetime.strptime(date_str, "%d %B, %Y")
        else:
            # Định dạng không năm (năm hiện tại): "21 June"
            current_year = datetime.now().year
            dt = datetime.strptime(f"{date_str} {current_year}", "%d %B %Y")
        return dt.strftime("%Y-%m-%d")
    except Exception as e:
        return None
        
# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        STEAM_IDS = [int(arg) for arg in sys.argv[1:]]
        
    print("[?] Using VPN is recommended because Steam is blocked in this region.")
    print(f"[*] Starting Steam Data Crawler for {len(STEAM_IDS)} target accounts...")
    
    for STEAM_ID in STEAM_IDS:
        print(f"\n\n=== STEAM DATA CRAWLER FOR {STEAM_ID} ===")
        
        # skip crawling if steam_id already exists in players.csv to avoid duplicates
        if is_already_crawled(STEAM_ID):
            print(f"[*] This STEAM_ID has already been crawled.")
            continue
            
        # 1. Players
        df_players = crawl_player_info(STEAM_ID)
        # if player info is None, it likely means the profile is private or invalid.
        if df_players is None:
            print("[-] Stopping further crawling due to private/invalid profile.")
            continue
        else:
            save_append(df_players, "players.csv")
        
            
        # 2. Purchased Games
        df_purchased, app_ids = crawl_library(STEAM_ID)
        if df_purchased is not None: save_append(df_purchased, "purchased_games.csv")
            
        # 3. History
        if app_ids:
            df_history = crawl_achievements(STEAM_ID, app_ids)
            if not df_history.empty:
                save_append(df_history, "history.csv")
                print(f"    -> Found {len(df_history)} unlocked achievements!")
            else:
                print("    -> No achievements found.")
                df_empty = pd.DataFrame(columns=["playerid", "achievementid", "date_accquired"])
                save_append(df_empty, "history.csv")
                
        # 5. Reviews (Đã nâng cấp)
        df_reviews = crawl_reviews(STEAM_ID)
        save_append(df_reviews, "reviews.csv")
    
        # 6. Private IDs (Vẫn giữ mock rỗng để AI Pipeline không lỗi)
        # save_append(pd.DataFrame(columns=["playerid"]), "private_steamids.csv")

        print(f"[v] Appended data for {STEAM_ID} to '{OUTPUT_DIR}'.\n")