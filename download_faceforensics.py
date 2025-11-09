#!/usr/bin/env python
"""
FaceForensics++ Download Script - With Auto Fallback
Downloads videos from FaceForensics++ official repository
Automatically tries alternate servers if primary fails
"""

import argparse
import os
import urllib.request
import urllib.error
import tempfile
import time
import sys
import json
from tqdm import tqdm
from pathlib import Path

# URLs and filenames
FILELIST_URL = 'misc/filelist.json'
DEEPFAKES_DETECTION_URL = 'misc/deepfake_detection_filenames.json'
DEEPFAKES_MODEL_NAMES = ['decoder_A.h5', 'decoder_B.h5', 'encoder.h5']

# Dataset paths
DATASETS = {
    'original_youtube_videos': 'misc/downloaded_youtube_videos.zip',
    'original_youtube_videos_info': 'misc/downloaded_youtube_videos_info.zip',
    'original': 'original_sequences/youtube',
    'DeepFakeDetection_original': 'original_sequences/actors',
    'Deepfakes': 'manipulated_sequences/Deepfakes',
    'DeepFakeDetection': 'manipulated_sequences/DeepFakeDetection',
    'Face2Face': 'manipulated_sequences/Face2Face',
    'FaceShifter': 'manipulated_sequences/FaceShifter',
    'FaceSwap': 'manipulated_sequences/FaceSwap',
    'NeuralTextures': 'manipulated_sequences/NeuralTextures'
}

ALL_DATASETS = [
    'original', 'DeepFakeDetection_original', 'Deepfakes', 'DeepFakeDetection',
    'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures'
]

COMPRESSION = ['raw', 'c23', 'c40']
TYPE = ['videos', 'masks', 'models']
SERVERS = ['EU', 'EU2', 'CA']


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Downloads FaceForensics++ public data release.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('output_path', type=str, help='Output directory')
    parser.add_argument('-d', '--dataset', type=str, default='all',
                       help='Which dataset to download',
                       choices=list(DATASETS.keys()) + ['all'])
    parser.add_argument('-c', '--compression', type=str, default='c23',
                       help='Compression level', choices=COMPRESSION)
    parser.add_argument('-t', '--type', type=str, default='videos',
                       help='File type to download', choices=TYPE)
    parser.add_argument('-n', '--num_videos', type=int, default=None,
                       help='Limit number of videos')
    parser.add_argument('--server', type=str, default='EU',
                       help='Download server', choices=SERVERS)
    
    args = parser.parse_args()
    return args


def get_server_urls(server):
    """Get URLs for specified server"""
    servers = {
        'EU': {
            'base': 'http://canis.vc.in.tum.de:8100/',
            'name': 'EU (TUM)'
        },
        'EU2': {
            'base': 'http://kaldir.vc.in.tum.de/faceforensics/',
            'name': 'EU2 (Alternative)'
        },
        'CA': {
            'base': 'http://falas.cmpt.sfu.ca:8100/',
            'name': 'CA (Canada)'
        }
    }
    
    if server not in servers:
        raise Exception(f'Unknown server: {server}')
    
    server_url = servers[server]['base']
    return {
        'server_name': servers[server]['name'],
        'tos_url': server_url + 'webpage/FaceForensics_TOS.pdf',
        'base_url': server_url + 'v3/',
        'deepfakes_model_url': server_url + 'v3/manipulated_sequences/Deepfakes/models/'
    }


def test_connection(base_url, timeout=5):
    """Test if server is reachable"""
    try:
        urllib.request.urlopen(base_url + FILELIST_URL, timeout=timeout)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, Exception):
        return False


def find_working_server(preferred_server, timeout=5):
    """Find working server, fallback to alternatives if needed"""
    print(f"\n[INFO] Testing server connection: {preferred_server}...")
    
    server_order = [preferred_server]
    # Add alternatives
    for s in SERVERS:
        if s != preferred_server:
            server_order.append(s)
    
    for server in server_order:
        urls = get_server_urls(server)
        print(f"  Trying {urls['server_name']}...", end='')
        
        if test_connection(urls['base_url'], timeout):
            print(" ✓ Connected!")
            return server, urls
        else:
            print(" ✗ Failed")
    
    print("\n✗ ERROR: All servers unreachable!")
    print("Possible reasons:")
    print("  - Network connectivity issue")
    print("  - All TUM servers are down")
    print("  - Firewall blocking connections")
    print("\n[ALTERNATIVE] Try downloading from Kaggle instead:")
    print("  kaggle datasets download -d xdxd003/ff-c23")
    sys.exit(1)


def download_file(url, out_file, report_progress=False):
    """Download a single file"""
    out_dir = os.path.dirname(out_file)
    
    if not os.path.isfile(out_file):
        os.makedirs(out_dir, exist_ok=True)
        fh, out_file_tmp = tempfile.mkstemp(dir=out_dir)
        f = os.fdopen(fh, 'w')
        f.close()
        
        try:
            if report_progress:
                urllib.request.urlretrieve(url, out_file_tmp, reporthook=reporthook)
            else:
                urllib.request.urlretrieve(url, out_file_tmp)
            
            os.rename(out_file_tmp, out_file)
        except Exception as e:
            print(f"\n✗ Download failed: {str(e)[:100]}")
            if os.path.exists(out_file_tmp):
                os.remove(out_file_tmp)
            raise
    else:
        tqdm.write(f'WARNING: Skipping existing file {out_file}')


def download_files(filenames, base_url, output_path, report_progress=True):
    """Download multiple files"""
    os.makedirs(output_path, exist_ok=True)
    
    if report_progress:
        filenames = tqdm(filenames, desc="Downloading files")
    
    for filename in filenames:
        download_file(base_url + filename, os.path.join(output_path, filename))


def reporthook(count, block_size, total_size):
    """Progress reporting for urllib downloads"""
    global start_time
    if count == 0:
        start_time = time.time()
        return
    
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration)) if duration > 0 else 0
    percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
    
    sys.stdout.write(
        f"\rProgress: {percent}%, {progress_size / (1024 * 1024):.0f} MB, "
        f"{speed} KB/s, {duration:.0f} seconds"
    )
    sys.stdout.flush()


def main(args):
    """Main download function"""
    print('=' * 80)
    print('FaceForensics++ Dataset Download (With Server Fallback)')
    print('=' * 80)
    print('\nBy pressing any key to continue you confirm that you have agreed')
    print('to the FaceForensics terms of use.')
    print('\n***')
    print('Press ENTER to continue, or CTRL-C to exit.')
    _ = input('')
    
    # Find working server
    working_server, urls = find_working_server(args.server)
    
    c_datasets = [args.dataset] if args.dataset != 'all' else ALL_DATASETS
    c_type = args.type
    c_compression = args.compression
    num_videos = args.num_videos
    output_path = args.output_path
    
    os.makedirs(output_path, exist_ok=True)
    
    # Download each dataset
    for dataset in c_datasets:
        dataset_path = DATASETS[dataset]
        
        # Special case: original youtube videos (zip file)
        if 'original_youtube_videos' in dataset:
            print(f'\nDownloading original youtube videos...')
            if 'info' not in dataset_path:
                print('Please be patient, this may take a while (~40GB)')
                suffix = ''
            else:
                suffix = 'info'
            
            download_file(
                urls['base_url'] + '/' + dataset_path,
                out_file=os.path.join(output_path, f'downloaded_videos{suffix}.zip'),
                report_progress=True
            )
            return
        
        # Regular datasets
        print(f'\n{"="*80}')
        print(f'Downloading {c_type} of dataset "{dataset_path}"')
        print(f'Server: {urls["server_name"]}')
        print(f'Compression: {c_compression}')
        print(f'{"="*80}')
        
        try:
            # Get filelist from server
            if 'DeepFakeDetection' in dataset_path or 'actors' in dataset_path:
                filepaths = json.loads(
                    urllib.request.urlopen(urls['base_url'] + '/' + DEEPFAKES_DETECTION_URL)
                    .read().decode("utf-8")
                )
                
                if 'actors' in dataset_path:
                    filelist = filepaths['actors']
                else:
                    filelist = filepaths['DeepFakesDetection']
            
            elif 'original' in dataset_path:
                file_pairs = json.loads(
                    urllib.request.urlopen(urls['base_url'] + '/' + FILELIST_URL)
                    .read().decode("utf-8")
                )
                filelist = []
                for pair in file_pairs:
                    filelist += pair
            
            else:
                file_pairs = json.loads(
                    urllib.request.urlopen(urls['base_url'] + '/' + FILELIST_URL)
                    .read().decode("utf-8")
                )
                filelist = []
                for pair in file_pairs:
                    filelist.append('_'.join(pair))
                    if c_type != 'models':
                        filelist.append('_'.join(pair[::-1]))
            
            # Limit number of videos if specified
            if num_videos is not None and num_videos > 0:
                print(f'Limiting to first {num_videos} videos')
                filelist = filelist[:num_videos]
            
            # Set up URLs and paths
            dataset_videos_url = urls['base_url'] + f'{dataset_path}/{c_compression}/{c_type}/'
            dataset_mask_url = urls['base_url'] + f'{dataset_path}/masks/{c_type}/'
            
            # Download based on type
            if c_type == 'videos':
                dataset_output_path = os.path.join(output_path, dataset_path, c_compression, c_type)
                print(f'Output path: {dataset_output_path}')
                filelist = [filename + '.mp4' for filename in filelist]
                download_files(filelist, dataset_videos_url, dataset_output_path)
            
            elif c_type == 'masks':
                dataset_output_path = os.path.join(output_path, dataset_path, c_type, 'videos')
                print(f'Output path: {dataset_output_path}')
                
                if 'original' in dataset:
                    print('Only videos available for original data. Skipping.')
                    continue
                
                if 'FaceShifter' in dataset:
                    print('Masks not available for FaceShifter. Skipping.')
                    continue
                
                filelist = [filename + '.mp4' for filename in filelist]
                download_files(filelist, dataset_mask_url, dataset_output_path)
            
            else:  # models
                if dataset != 'Deepfakes' and c_type == 'models':
                    print('Models only available for Deepfakes. Skipping.')
                    continue
                
                dataset_output_path = os.path.join(output_path, dataset_path, c_type)
                print(f'Output path: {dataset_output_path}')
                
                for folder in tqdm(filelist, desc="Downloading models"):
                    folder_base_url = urls['deepfakes_model_url'] + folder + '/'
                    folder_output_path = os.path.join(dataset_output_path, folder)
                    download_files(
                        DEEPFAKES_MODEL_NAMES,
                        folder_base_url,
                        folder_output_path,
                        report_progress=False
                    )
        
        except Exception as e:
            print(f"\n✗ Error downloading {dataset}: {str(e)[:100]}")
            print("Continuing with next dataset...")
            continue
    
    print(f'\n{"="*80}')
    print('✓ Download complete!')
    print(f'{"="*80}')
    print(f'Output directory: {output_path}')


if __name__ == "__main__":
    args = parse_args()
    main(args)