import os
import urllib.request

# The exact folder where the emnist library looks for cached data
cache_dir = os.path.expanduser('~/.cache/emnist')
os.makedirs(cache_dir, exist_ok=True)
zip_path = os.path.join(cache_dir, 'emnist.zip')

# The updated, working NIST server link
url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"

def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = downloaded * 100 / total_size
        if percent > 100: percent = 100
        print(f"\rDownloading: {percent:.1f}%", end="")

print("Downloading the EMNIST dataset from the updated server...")
print("This is a 536 MB file. Please do not close the terminal.")

# Adding a User-Agent so the government server doesn't block the download
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

# Download the file to the cache
urllib.request.urlretrieve(url, zip_path, show_progress)

print(f"\n\nSuccess! The real dataset is saved to: {zip_path}")
print("The broken library is officially bypassed.")