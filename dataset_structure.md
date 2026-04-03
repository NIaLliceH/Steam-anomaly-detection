# Steam data
## achievements (144.82 MB)
- achievementid: gameid_name
- gameid
- title
- description

## friends (658.13 MB)
- playerid
- friends: list of playerid

## games (17.07 MB)
- gameid
- title
- developers: list of name (mostly one item)
- publishers: list of name (mostly one item)
- genres: list of genre
- supported_languages: list of lang
- release_date: YYYY-MM-DD

## history (647.56 MB)
- playerid
- achievementid
- date_acquired: YYYY-MM-DD HH-MM-SS

## players (18.41 MB)
- playerid
- country: where user live
- created: YYYY-MM-DD HH-MM-SS

## prices (183.92 MB)
- gameid
- usd: America
- eur: Europe
- gbp: England
- jpy: Japan
- rub: Russia
- date_acquired: YYYY-MM-DD

## private_steamids (4.33 MB)
- playerid

## purchased_games (91.73 MB)
- playerid
- library: list of gameid

## reviews (551.64 MB)
- reviewid
- playerid
- gameid
- review: text in different lang
- helpful: number of like for this review (mostly zero)
- funny: (mostly zero)
- awards: (mostly zero)
- posted: YYYY-MM-DD