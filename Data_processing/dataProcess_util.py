import pandas as pd
import requests
import urllib.parse
import os
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def extract_actual_url(url):
    """
    Extract the actual image URL from a Reddit media redirect URL.
    """
    if 'reddit.com/media?url=' in url:
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        if 'url' in params:
            return params['url'][0]
    return url

def download_image_improved(url, save_path, timeout=10):
    """
    Args:
        url: Image URL
        save_path: Path to save the image
        timeout: Request timeout in seconds

    Returns:
        tuple: (success: bool, error_msg: str or None)
    """
    try:
        actual_url = extract_actual_url(url)

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(actual_url, headers=headers, timeout=timeout, stream=True)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            return False, f"Not an image (Content-Type: {content_type})"

        # Try to open and save the image
        img = Image.open(BytesIO(response.content))

        # Convert RGBA to RGB if necessary (for JPEG compatibility)
        if img.mode in ('RGBA', 'LA', 'P'):
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            rgb_img.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = rgb_img

        img.save(save_path, 'JPEG', quality=95)
        return True, None

    except requests.exceptions.Timeout:
        return False, "Timeout"
    except requests.exceptions.HTTPError as e:
        return False, f"HTTP {e.response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Connection error"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)}"

def download_images_batch_improved(df, save_dir='fakeddit/images', max_images=None,
                                  num_workers=5, failed_log_path=None):
    """
    Batch download images with detailed failure tracking.

    Args:
        df: DataFrame with image_url column
        save_dir: Directory to save images
        max_images: Maximum number of images to download (None means all)
        num_workers: Number of parallel workers
        failed_log_path: Path to save failed downloads log (TSV format).
                        If None, defaults to '{save_dir}/failed_downloads.tsv'

    Returns:
        tuple: (successful_count, failed_df)
            - successful_count: Number of successfully downloaded images
            - failed_df: DataFrame with failed downloads (columns: original_index, id, image_url, error_reason)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Set default log path if not provided
    if failed_log_path is None:
        failed_log_path = os.path.join(save_dir, 'failed_downloads.tsv')

    valid_df = df[df['image_url'].notna()].copy()
    if max_images:
        valid_df = valid_df.head(max_images)

    print(f"START DOWNLOADING {len(valid_df)} images...")

    # Prepare download tasks with metadata
    download_tasks = []
    for idx, row in valid_df.iterrows():
        url = row['image_url']
        if 'id' in row and pd.notna(row['id']):
            file_name = f"{row['id']}.jpg"
            row_id = row['id']
        else:
            file_name = f"{idx}.jpg"
            row_id = str(idx)
        save_path = os.path.join(save_dir, file_name)
        download_tasks.append((idx, row_id, url, save_path))

    # Track failures
    failed_records = []
    successful = 0

    def download_with_metadata(task):
        idx, row_id, url, save_path = task
        success, error_msg = download_image_improved(url, save_path)
        return (idx, row_id, url, success, error_msg)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(download_with_metadata, download_tasks),
            total=len(download_tasks),
            desc="DOWNLOAD PROGRESS"
        ))

    # Process results
    for idx, row_id, url, success, error_msg in results:
        if success:
            successful += 1
        else:
            failed_records.append({
                'original_index': idx,
                'id': row_id,
                'image_url': url,
                'error_reason': error_msg
            })

    # Create failed DataFrame and save to file
    failed_df = pd.DataFrame(failed_records)
    if len(failed_df) > 0:
        failed_df.to_csv(failed_log_path, sep='\t', index=False)
        print(f"\nFailed downloads logged to: {failed_log_path}")

    print(f"\nDownload Complete!")
    print(f"  Successful: {successful}/{len(download_tasks)}")
    print(f"  Failed: {len(failed_records)}/{len(download_tasks)}")

    if len(failed_records) > 0:
        # Print failure summary
        error_summary = failed_df['error_reason'].value_counts()
        print(f"\nFailure reasons:")
        for reason, count in error_summary.items():
            print(f"  - {reason}: {count}")

    return successful, failed_df

def filter_failed_rows(original_df, failed_df):
    """
    Remove rows from the original DataFrame that failed to download.

    Args:
        original_df: Original DataFrame
        failed_df: DataFrame returned by download_images_batch_improved containing failed downloads

    Returns:
        DataFrame: Filtered DataFrame with successful downloads only
    """
    if len(failed_df) == 0:
        print("No failed downloads to filter out.")
        return original_df

    # Get indices of failed downloads
    failed_indices = set(failed_df['original_index'].tolist())

    # Filter out failed rows
    filtered_df = original_df[~original_df.index.isin(failed_indices)].copy()

    print(f"\nFiltered out {len(failed_indices)} failed downloads.")
    print(f"Remaining rows: {len(filtered_df)}")

    return filtered_df