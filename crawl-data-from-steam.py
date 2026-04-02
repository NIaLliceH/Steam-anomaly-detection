import sys
import requests
import pandas as pd
from datetime import datetime
import time
import os
from dotenv import load_dotenv

# ==========================================
# CONFIGURATION
# ==========================================

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")
STEAM_ID = "76561198287996067"  # Điền SteamID64 (chuỗi 17 chữ số)
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

def mock_empty_reviews(steam_id):
    print("[+] Mocking empty reviews (No official per-user API available)...")
    # Tạo bảng rỗng với đúng tên cột để Pipeline Step 1 không bị crash
    return pd.DataFrame(columns=[
        "reviewid", "playerid", "gameid", "review", 
        "helpful", "funny", "awards", "posted"
    ])
    
def is_already_crawled(steam_id):
    """Kiểm tra xem STEAM_ID đã tồn tại trong file players.csv chưa"""
    filepath = os.path.join(OUTPUT_DIR, "players.csv")
    if not os.path.exists(filepath):
        return False
    
    try:
        # Chỉ đọc đúng cột playerid để tiết kiệm RAM
        df = pd.read_csv(filepath, usecols=["playerid"], dtype={"playerid": str})
        return str(steam_id) in df["playerid"].values
    except Exception as e:
        print(f"[!] Warning checking existence: {e}")
        return False

def save_append(df, filename):
    """Hàm lưu nối tiếp vào CSV. Tự động tạo header nếu file chưa tồn tại."""
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        df.to_csv(filepath, index=False)
    else:
        df.to_csv(filepath, mode='a', header=False, index=False)
        
# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        STEAM_ID = sys.argv[1]
    
    print(f"=== STEAM DATA CRAWLER FOR {STEAM_ID} ===")
    
    # skip crawling if steam_id already exists in players.csv to avoid duplicates
    if is_already_crawled(STEAM_ID):
        print(f"[*] This STEAM_ID has already been crawled.")
        sys.exit(0)
        
    # 1. Players
    df_players = crawl_player_info(STEAM_ID)
    # if player info is None, it likely means the profile is private or invalid.
    if df_players is None:
        print("[-] Stopping further crawling due to private/invalid profile.")
        sys.exit(0)
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
            
    # 4. Reviews & Private IDs (Mock)
    save_append(mock_empty_reviews(STEAM_ID), "reviews.csv")
    save_append(pd.DataFrame(columns=["playerid"]), "private_steamids.csv")

    print(f"[v] Appended data for {STEAM_ID} to '{OUTPUT_DIR}'.\n")