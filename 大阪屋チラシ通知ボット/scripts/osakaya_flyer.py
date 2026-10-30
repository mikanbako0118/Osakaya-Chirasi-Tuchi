#!/usr/bin/env python3
"""
大阪屋ショップ 上北島店のチラシから「本当に安い商品」を抜き出して Discord に投稿する。

処理の流れ:
  1. トクバイの「最新チラシへのリダイレクトURL」を叩いて、今出ているチラシのページを取得
  2. そのHTMLからチラシ画像のID(数字)を全部拾い、原寸画像URLに組み立てる
  3. 画像をダウンロードし、そのままだと文字が小さすぎるので 2x2 のタイルに切り分ける
  4. 画像を Gemini に渡して、特売情報を読み取ってMarkdownに整形させる
  5. できた文章を Discord の Webhook に投げる(2000文字を超えたら分割)

必要な環境変数:
  GEMINI_API_KEY       ... Google AI Studio で取得したAPIキー(無料枠で動きます)
  DISCORD_WEBHOOK_URL  ... Discord の Webhook URL
"""

from __future__ import annotations

import io
import os
import re
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from PIL import Image
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

SHOP_ID = "170244"                      # トクバイの店舗ID(上北島店)
SHOP_NAME = "大阪屋ショップ 上北島店"

# このURLは常に「今出ている最新チラシ」のページにリダイレクトされる
REDIRECT_URL = (
    f"https://tokubai.co.jp/{SHOP_ID}/redirects/leaflet_page_link"
    "?from=widget_external_link"
)

# 無料枠で使えるモデルを上から順に試す。
# (Googleがモデル名を変えても、どれかが通れば動き続けるようにしてある)
MODEL_CANDIDATES = [
    "gemini-3-flash",
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-2.0-flash",
]
MAX_OUTPUT_TOKENS = 4000

# 画像は長辺 1568px 以下にしておくと、どのモデルでも縮小されずに読まれる。
# チラシ原寸は 2236x1586 程度なので、2x2 に切ると 1タイル約 1185x840 で収まる。
TILE_COLS = 2
TILE_ROWS = 2
TILE_OVERLAP = 0.06        # タイルの境界で商品が切れないよう 6% 重ねる
OVERVIEW_MAX_EDGE = 1200   # 全体像(俯瞰用)はここまで縮小

DISCORD_CHUNK_LIMIT = 1900  # Discord の上限は2000文字。余裕を持たせる

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

JST = ZoneInfo("Asia/Tokyo")


# ---------------------------------------------------------------------------
# 1. チラシページのHTMLを取る
# ---------------------------------------------------------------------------

def fetch_leaflet_html() -> tuple[str, str]:
    """最新チラシページのHTMLと最終URLを返す。"""
    res = requests.get(
        REDIRECT_URL,
        headers={"User-Agent": USER_AGENT},
        timeout=30,
        allow_redirects=True,
    )
    res.raise_for_status()
    res.encoding = res.apparent_encoding or "utf-8"
    return res.text, res.url


# ---------------------------------------------------------------------------
# 2. HTMLからチラシ画像のURLを組み立てる
# ---------------------------------------------------------------------------

# HTML中の画像URLは2種類の形で出てくる:
#   .../bargain_office_leaflets/o=true/9536077.jpg              (原寸)
#   .../bargain_office_leaflets/w=180,h=135,c=true/9536081.jpg  (サムネイル)
# どちらからも末尾の数字(画像ID)だけを取り出したい。
IMAGE_ID_RE = re.compile(r"bargain_office_leaflets/[^\"'\s>]*?(\d{6,})\.jpg")


def extract_image_urls(html: str) -> list[str]:
    """HTMLからチラシ画像の原寸URL一覧を、重複を除いて出現順に返す。"""
    urls: list[str] = []
    seen: set[str] = set()
    for image_id in IMAGE_ID_RE.findall(html):
        if image_id in seen:
            continue
        seen.add(image_id)
        urls.append(
            "https://image.tokubai.co.jp/images/"
            f"bargain_office_leaflets/o=true/{image_id}.jpg"
        )
    return urls


# ---------------------------------------------------------------------------
# 3. 画像をダウンロードしてタイルに切る
# ---------------------------------------------------------------------------

def download_image(url: str) -> Image.Image:
    res = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=90)
    res.raise_for_status()
    return Image.open(io.BytesIO(res.content)).convert("RGB")


def shrink(img: Image.Image, max_edge: int) -> Image.Image:
    """長辺が max_edge を超えていたら縮小する。超えていなければそのまま。"""
    if max(img.size) <= max_edge:
        return img
    scale = max_edge / max(img.size)
    return img.resize(
        (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
        Image.LANCZOS,
    )


def split_into_tiles(img: Image.Image) -> list[Image.Image]:
    """画像を TILE_COLS x TILE_ROWS に分割する(境界は少し重ねる)。"""
    width, height = img.size
    tile_w = width / TILE_COLS
    tile_h = height / TILE_ROWS
    pad_x = tile_w * TILE_OVERLAP
    pad_y = tile_h * TILE_OVERLAP

    tiles: list[Image.Image] = []
    for row in range(TILE_ROWS):
        for col in range(TILE_COLS):
            left = max(0, int(col * tile_w - pad_x))
            top = max(0, int(row * tile_h - pad_y))
            right = min(width, int((col + 1) * tile_w + pad_x))
            bottom = min(height, int((row + 1) * tile_h + pad_y))
            tiles.append(shrink(img.crop((left, top, right, bottom)), 1568))
    return tiles


def to_part(img: Image.Image) -> types.Part:
    """Pillow の画像を Gemini API に渡せる形(Part)に変換する。"""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88, optimize=True)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


# ---------------------------------------------------------------------------
# 4. AIに読ませる
# ---------------------------------------------------------------------------

TILE_LABELS = ["左上", "右上", "左下", "右下"]

PROMPT_HEADER = """\
あなたはスーパーのチラシを読んで、本当にお得な商品だけを抜き出す担当者です。

今日の日付: {today}

これから「{shop}」のチラシ画像を渡します。
チラシは1〜2ページあり、各ページについて次の順で渡します:
  (a) ページ全体の俯瞰画像 1枚 … レイアウトと日付の見出しを把握するため
  (b) 拡大タイル4枚(左上・右上・左下・右下の順、境界は少し重なっています) … 価格を正確に読むため

## 最重要: 日付の判定
チラシには「本日7/28(火)限り」「7/29(水)限り」「7/28〜31 4日間連日」など、
適用日が違うブロックが混在しています。
**今日({today})に買えないものは絶対に載せないでください。**
- 大きな日付の見出し(例:「7/30(木)限り」)は、その見出しが囲む枠やその下の列全体に効きます。
  個々の商品に日付が書かれていなくても、見出しの範囲内なら同じ日付が適用されます。
- 期間表示(例:「7/28・29・30・31 4日間連日」)は、今日がその範囲内なら有効です。
- 判断に迷う場合は「載せない」を選んでください。

## 拾う基準(本当に安いものだけ)
- 「半額」と書かれているもの
- 「4割引」「6割引」「30%引」など値引き率が大きいもの
- 「本日限り」などで単品に大きく赤字価格が出ている目玉商品
- 精肉・鮮魚・青果で、明らかに安い目玉価格のもの

## 除外するもの
- カテゴリ一括の薄い割引(「◯◯各種10%引」「20%引」など、商品が特定されないもの)
- 今日が対象外の日付のブロック(上記参照)
- 会員属性による割引(シルバーデーなど)は本文に入れず、最後の注記に1行だけ書く

## 価格の書き方
税抜と税込が併記されている場合は**税込価格**を使ってください。

## 出力フォーマット
以下の形式のMarkdownだけを出力してください。前置きや説明は書かないこと。
該当がないカテゴリは省略してよいですが、「半額・大幅引き」に該当がない場合は
「・該当なし」と書き、理由(期限切れなど)を分かる範囲で添えてください。

```
🔥 {shop} 特売(チラシに記載の日付範囲)

**半額・大幅引き**
・商品名 — 割引内容

**本日限り目玉〈{today_short}〉**
・商品名 価格(税込)〈個数制限があれば〉

**生鮮**
▼鮮魚
・商品名 価格(税込)
▼精肉
・商品名 価格(税込)
▼青果
・商品名 価格(税込)
▼惣菜
・商品名 価格(税込)
```

全体で1800文字以内に収めてください。安い順・目玉順に並べ、似た商品は1行にまとめて構いません。
"""


def build_contents(pages: list[Image.Image], today: datetime) -> list:
    """Gemini に渡す contents(テキストと画像を交互に並べたリスト)を組み立てる。"""
    weekdays = "月火水木金土日"
    today_str = f"{today.year}年{today.month}月{today.day}日({weekdays[today.weekday()]})"
    today_short = f"{today.month}/{today.day}({weekdays[today.weekday()]})"

    contents: list = [
        PROMPT_HEADER.format(today=today_str, today_short=today_short, shop=SHOP_NAME)
    ]

    for page_no, page in enumerate(pages, start=1):
        contents.append(f"=== {page_no}ページ目: 全体の俯瞰 ===")
        contents.append(to_part(shrink(page, OVERVIEW_MAX_EDGE)))
        for label, tile in zip(TILE_LABELS, split_into_tiles(page)):
            contents.append(f"--- {page_no}ページ目 拡大: {label} ---")
            contents.append(to_part(tile))

    return contents


def ask_ai(api_key: str, contents: list) -> str:
    """モデル候補を順に試して、最初に成功したものの結果を返す。"""
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=0,   # 毎回ぶれないように
    )

    errors: list[str] = []
    for model in MODEL_CANDIDATES:
        for attempt in range(3):
            try:
                res = client.models.generate_content(
                    model=model, contents=contents, config=config
                )
                text = (res.text or "").strip()
                if not text:
                    raise RuntimeError("空の応答が返りました")
                print(f"        使用モデル: {model}")
                return strip_code_fence(text)
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {exc}"
                lowered = msg.lower()
                # レート制限・一時的な混雑は待って再試行
                if any(k in lowered for k in ("429", "resource_exhausted", "503", "unavailable")):
                    wait = 20 * (attempt + 1)
                    print(f"        {model}: 混雑のため{wait}秒待って再試行 ({msg[:120]})")
                    time.sleep(wait)
                    continue
                # モデル名が無い等は次の候補へ
                errors.append(f"{model} -> {msg[:200]}")
                break
        else:
            errors.append(f"{model} -> 再試行しても混雑が解消しませんでした")

    raise RuntimeError("すべてのモデルで失敗しました:\n" + "\n".join(errors))


def strip_code_fence(text: str) -> str:
    """``` で囲んで返ってきた場合に外側のフェンスを外す。"""
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


# ---------------------------------------------------------------------------
# 5. Discord に投稿
# ---------------------------------------------------------------------------

def split_for_discord(text: str, limit: int = DISCORD_CHUNK_LIMIT) -> list[str]:
    """Discord の文字数上限に合わせて、行の途中で切らないように分割する。"""
    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        # 1行だけで上限を超える異常ケースは強制的に切る
        while len(line) > limit:
            if current:
                chunks.append(current.rstrip())
                current = ""
            chunks.append(line[:limit])
            line = line[limit:]
        if len(current) + len(line) + 1 > limit:
            chunks.append(current.rstrip())
            current = line + "\n"
        else:
            current += line + "\n"
    if current.strip():
        chunks.append(current.rstrip())
    return chunks or [text]


def post_to_discord(webhook_url: str, text: str) -> None:
    for i, chunk in enumerate(split_for_discord(text)):
        res = requests.post(
            webhook_url,
            json={"content": chunk},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        if res.status_code == 429:  # レート制限。待ってから1回だけ再送
            wait = float(res.json().get("retry_after", 2))
            time.sleep(wait + 0.5)
            res = requests.post(webhook_url, json={"content": chunk}, timeout=30)
        res.raise_for_status()
        if i > 0:
            time.sleep(1)  # 連投しすぎない


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        sys.exit(f"環境変数 {name} が設定されていません。")
    return value


def main() -> None:
    api_key = require_env("GEMINI_API_KEY")
    webhook_url = require_env("DISCORD_WEBHOOK_URL")

    try:
        today = datetime.now(JST)
        print(f"[1/5] 実行日(JST): {today:%Y-%m-%d %H:%M}")

        html, final_url = fetch_leaflet_html()
        print(f"[2/5] チラシページ取得: {final_url}")

        image_urls = extract_image_urls(html)
        if not image_urls:
            raise RuntimeError(
                "チラシ画像のURLが1件も見つかりませんでした。"
                "サイトのHTML構造が変わった可能性があります。"
            )
        print(f"[3/5] チラシ画像 {len(image_urls)}枚: " + ", ".join(image_urls))

        pages = [download_image(u) for u in image_urls]
        for u, p in zip(image_urls, pages):
            print(f"        {u.rsplit('/', 1)[-1]} -> {p.width}x{p.height}")

        contents = build_contents(pages, today)
        n_images = sum(1 for c in contents if not isinstance(c, str))
        print(f"[4/5] Gemini に問い合わせ中(画像 {n_images}枚)...")
        message = ask_ai(api_key, contents)
        print("--- 生成されたメッセージ ---")
        print(message)
        print("---------------------------")

        post_to_discord(webhook_url, message)
        print("[5/5] Discord に投稿しました。")

    except Exception as exc:  # noqa: BLE001
        # 失敗しても無言で終わらせず、Discord にエラーを通知する
        detail = f"{type(exc).__name__}: {exc}"
        print(f"エラー: {detail}", file=sys.stderr)
        try:
            post_to_discord(
                webhook_url,
                f"⚠️ {SHOP_NAME} のチラシ取得に失敗しました。\n```\n{detail[:1500]}\n```",
            )
        except Exception as notify_error:  # noqa: BLE001
            print(f"エラー通知も失敗: {notify_error}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
