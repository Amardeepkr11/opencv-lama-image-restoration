"""Download the 24 Kodak Lossless True Color images (kodim01..kodim24.png) for the benchmark.

usage: .\\venv\\Scripts\\python.exe get_photos.py [--out samples/benchmark]
"""
import argparse
import os
import time
import urllib.request

SOURCES = [
    "http://r0k.us/graphics/kodak/kodak/kodim{nn}.png",
    "https://raw.githubusercontent.com/MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite/master/PhotoCD_PCD0992/{n}.png",
]
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) inpainting-benchmark/1.0"}
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def fetch(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def main():
    parser = argparse.ArgumentParser(description="Download the Kodak image set.")
    parser.add_argument("--out", default=os.path.join("samples", "benchmark"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ok = 0
    for i in range(1, 25):
        name = f"kodim{i:02d}.png"
        path = os.path.join(args.out, name)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"[=] {name} already present")
            ok += 1
            continue
        for template in SOURCES:
            url = template.format(nn=f"{i:02d}", n=i)
            try:
                data = fetch(url)
                if not data.startswith(PNG_MAGIC):
                    raise ValueError("response is not a PNG file")
                with open(path, "wb") as fh:
                    fh.write(data)
                print(f"[+] {name}  {len(data) // 1024} KB  from {url}")
                ok += 1
                break
            except Exception as e:
                print(f"[!] {name} failed from {url}: {e}")
                time.sleep(1)
        else:
            print(f"[x] {name} could not be downloaded from any source")

    print(f"\n{ok} of 24 Kodak images available in {args.out}")


if __name__ == "__main__":
    main()
