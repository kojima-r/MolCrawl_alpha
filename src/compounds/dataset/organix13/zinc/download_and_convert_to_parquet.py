import subprocess
import requests
from urllib.parse import urljoin
import time

import os.path as osp
import os
import shutil
import sys

import logging
import logging.config

# Add project root to Python path when running as script
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..", "..", "..", "..", "..")
    project_root = os.path.abspath(project_root)
    sys.path.insert(0, project_root)
    sys.path.insert(0, os.path.join(project_root, "src"))

from config.paths import COMPOUNDS_DIR

logger = logging.getLogger(__name__)


def generate_zinc_file_list():
    """
    Generate list of ZINC20 files to download.
    Based on the pattern from download_zinc.sh: 4-character combinations using A-K for first char, A-B for second char
    """
    files = []
    
    # Extract actual file list from the existing script to ensure compatibility
    try:
        shell_file = "src/compounds/dataset/organix13/zinc/zinc_complete/download_zinc.sh"
        if os.path.exists(shell_file):
            with open(shell_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if "wget" in line and ".txt" in line:
                        # Extract filename from wget command
                        parts = line.split()
                        for part in parts:
                            if part.endswith(".txt") and "/" in part:
                                filename = part.split("/")[-1]
                                if filename not in [f["filename"] for f in files]:
                                    # Extract directory from the mkdir command
                                    dir_name = filename[:2]  # First two characters
                                    files.append({
                                        "filename": filename,
                                        "directory": dir_name,
                                        "url": f"https://files.docking.org/2D/{dir_name}/{filename}"
                                    })
                                break
    except Exception as e:
        logger.warning(f"Could not read shell script, falling back to generated list: {e}")
        
        # Fallback: generate based on observed patterns
        # First character: A-K, Second character: A-B, Third and fourth: A,B,C,D,E
        first_chars = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
        second_chars = ['A', 'B']
        third_fourth_chars = ['A', 'B', 'C', 'D', 'E']
        
        for first in first_chars:
            for second in second_chars:
                for third in third_fourth_chars:
                    for fourth in third_fourth_chars:
                        filename = f"{first}{second}{third}{fourth}.txt"
                        dir_name = f"{first}{second}"
                        files.append({
                            "filename": filename,
                            "directory": dir_name,
                            "url": f"https://files.docking.org/2D/{dir_name}/{filename}"
                        })
    
    logger.info(f"Generated {len(files)} ZINC files for download")
    return files


def download_single_file(file_info, base_directory, retries=5, timeout=60):
    """
    Download a single ZINC file with retry logic.
    
    Args:
        file_info: Dict with 'filename', 'directory', 'url'
        base_directory: Base directory to save files
        retries: Number of retry attempts
        timeout: Timeout in seconds for each request
    """
    url = file_info["url"]
    dir_name = file_info["directory"]
    filename = file_info["filename"]
    
    # Create directory structure
    target_dir = osp.join(base_directory, dir_name)
    os.makedirs(target_dir, exist_ok=True)
    
    target_path = osp.join(target_dir, filename)
    
    # Skip if file already exists and is non-empty
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        logger.debug(f"File {target_path} already exists, skipping")
        return True
    
    for attempt in range(retries):
        try:
            logger.info(f"Downloading {url} (attempt {attempt + 1}/{retries})")
            
            # Use session for better connection handling
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            
            response = session.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Write file in chunks
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)
            
            session.close()
            
            # Verify file was downloaded successfully
            if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
                logger.info(f"Successfully downloaded {filename}")
                return True
            else:
                logger.warning(f"Downloaded file {filename} is empty")
                
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed for {filename}: {e}")
            if attempt < retries - 1:
                # Longer delay for server errors (503, 429, etc.)
                if hasattr(e, 'response') and e.response is not None and e.response.status_code in [503, 429, 502, 504]:
                    delay = min(30, 5 * (2 ** attempt))  # Cap at 30 seconds
                    logger.info(f"Server error detected, waiting {delay} seconds before retry...")
                    time.sleep(delay)
                else:
                    time.sleep(2 ** attempt)  # Exponential backoff for other errors
            else:
                logger.error(f"Failed to download {filename} after {retries} attempts")
                return False
        except Exception as e:
            logger.error(f"Unexpected error downloading {filename}: {e}")
            return False
    
    return False


def download_zinc_files(delay_between_downloads: float = 1.0):
    """
    Download ZINC20 files using Python requests with sequential processing.
    
    Args:
        delay_between_downloads: Delay in seconds between downloads to avoid 503 errors
    """
    directory = osp.join(COMPOUNDS_DIR, "zinc20")
    os.makedirs(directory, exist_ok=True)
    
    # Generate file list
    files_to_download = generate_zinc_file_list()
    
    logger.info(f"Starting sequential download of {len(files_to_download)} ZINC files to {directory}")
    logger.info(f"Using delay of {delay_between_downloads} seconds between downloads")
    
    successful_downloads = 0
    failed_downloads = 0
    
    # Download files sequentially to avoid 503 errors
    for i, file_info in enumerate(files_to_download):
        logger.info(f"Progress: {i+1}/{len(files_to_download)} - Downloading {file_info['filename']}")
        
        try:
            success = download_single_file(file_info, directory)
            if success:
                successful_downloads += 1
            else:
                failed_downloads += 1
        except Exception as e:
            logger.error(f"Error processing {file_info['filename']}: {e}")
            failed_downloads += 1
        
        # Add delay between downloads to avoid overwhelming the server
        if i < len(files_to_download) - 1:  # Don't delay after the last file
            time.sleep(delay_between_downloads)
    
    logger.info(f"ZINC downloads completed: {successful_downloads} successful, {failed_downloads} failed")
    
    if failed_downloads > 0:
        logger.warning(f"{failed_downloads} files failed to download. You may want to retry.")
    
    return successful_downloads, failed_downloads


def convert_zinc_to_parquet(save_path: str):
    """
    Convert downloaded ZINC files to parquet format.
    
    Args:
        save_path: Directory to save the final parquet file
    """
    # Import dask only when needed for parquet conversion
    try:
        import dask.dataframe as dd
    except ImportError:
        logger.error("dask is required for parquet conversion. Install with: pip install dask[dataframe]")
        return None
    
    base_directory = osp.join(COMPOUNDS_DIR, "zinc20")
    
    if not os.path.exists(base_directory):
        logger.error(f"ZINC directory {base_directory} does not exist. Run download_zinc_files first.")
        return
    
    # Find all directories containing .txt files
    all_dirs = [
        osp.join(base_directory, f)
        for f in os.listdir(base_directory)
        if osp.isdir(osp.join(base_directory, f))
    ]

    logger.info(f"Found {len(all_dirs)} directories in {base_directory}")
    
    all_dfs = []
    processed_files = 0
    failed_files = 0
    
    for dir_path in all_dirs:
        logger.info(f"Processing directory: {dir_path}")
        
        # Get all .txt files in this directory
        txt_files = [f for f in os.listdir(dir_path) if f.endswith('.txt')]
        
        if not txt_files:
            logger.warning(f"No .txt files found in {dir_path}")
            continue
            
        try:
            # Read all .txt files in this directory
            df = dd.read_csv(
                f"{dir_path}/*.txt",
                sep="\t",
                usecols=["smiles"],
            )
            all_dfs.append(df)
            processed_files += len(txt_files)
            logger.info(f"Successfully processed {len(txt_files)} files from {dir_path}")
            
        except Exception as e:
            logger.error(f"Error reading files from {dir_path}: {e}")
            failed_files += len(txt_files)
            continue

    if not all_dfs:
        logger.error("No valid dataframes found. Cannot create parquet file.")
        return

    logger.info(f"Concatenating {len(all_dfs)} dataframes...")
    concatenated_df = dd.concat(all_dfs)

    # Create output directory
    os.makedirs(save_path, exist_ok=True)
    
    logger.info("Writing parquet file...")
    concatenated_df = concatenated_df.repartition(npartitions=1)
    concatenated_df = concatenated_df.reset_index(drop=True)
    
    # Create temporary directory for parquet output
    temp_parquet_dir = osp.join(base_directory, "zinc_processed")
    concatenated_df.to_parquet(temp_parquet_dir)
    
    # Copy the single parquet file to final destination
    final_parquet_path = osp.join(save_path, "zinc_processed.parquet")
    shutil.copy(
        osp.join(temp_parquet_dir, "part.0.parquet"),
        final_parquet_path
    )
    
    # Clean up temporary directory
    if os.path.exists(temp_parquet_dir):
        shutil.rmtree(temp_parquet_dir)
    
    logger.info(f"Successfully created parquet file: {final_parquet_path}")
    logger.info(f"Processed {processed_files} files, {failed_files} files failed")
    
    return final_parquet_path


def check_download_status():
    """
    Check the status of ZINC downloads and provide a summary.
    
    Returns:
        Dict with download statistics
    """
    base_directory = osp.join(COMPOUNDS_DIR, "zinc20")
    
    if not os.path.exists(base_directory):
        return {
            "status": "not_started",
            "total_expected": 300,
            "downloaded": 0,
            "missing": 300,
            "empty_files": 0
        }
    
    files_to_check = generate_zinc_file_list()
    downloaded = 0
    missing = 0
    empty_files = 0
    
    for file_info in files_to_check:
        file_path = osp.join(base_directory, file_info["directory"], file_info["filename"])
        
        if os.path.exists(file_path):
            if os.path.getsize(file_path) > 0:
                downloaded += 1
            else:
                empty_files += 1
        else:
            missing += 1
    
    total_expected = len(files_to_check)
    completion_rate = (downloaded / total_expected) * 100 if total_expected > 0 else 0
    
    status = "complete" if downloaded == total_expected else "partial" if downloaded > 0 else "not_started"
    
    return {
        "status": status,
        "total_expected": total_expected,
        "downloaded": downloaded,
        "missing": missing,
        "empty_files": empty_files,
        "completion_rate": completion_rate,
        "base_directory": base_directory
    }


def main():
    """
    Main function to demonstrate usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and process ZINC20 data")
    parser.add_argument("--download", action="store_true", help="Download ZINC files")
    parser.add_argument("--convert", type=str, help="Convert to parquet and save to specified directory")
    parser.add_argument("--status", action="store_true", help="Check download status")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay in seconds between downloads (default: 1.0)")
    
    args = parser.parse_args()
    
    if args.status:
        status = check_download_status()
        print(f"Download Status: {status['status']}")
        print(f"Downloaded: {status['downloaded']}/{status['total_expected']} ({status['completion_rate']:.1f}%)")
        print(f"Missing: {status['missing']}")
        print(f"Empty files: {status['empty_files']}")
        print(f"Base directory: {status['base_directory']}")
    
    if args.download:
        print(f"Starting sequential download with {args.delay} second delay between downloads...")
        successful, failed = download_zinc_files(delay_between_downloads=args.delay)
        print(f"Download completed: {successful} successful, {failed} failed")
    
    if args.convert:
        print(f"Converting ZINC data to parquet format...")
        result = convert_zinc_to_parquet(args.convert)
        if result:
            print(f"Conversion completed: {result}")
        else:
            print("Conversion failed")


if __name__ == "__main__":
    main()
