#!/usr/bin/env python3
"""
무거운 데이터 파일 정리 스크립트
Cursor 성능 최적화용
"""

import os
import shutil
from pathlib import Path

def cleanup_heavy_files():
    """무거운 파일들을 별도 위치로 이동"""
    
    # 백업 디렉토리
    backup_dir = Path("data_backup")
    backup_dir.mkdir(exist_ok=True)
    
    # 이동할 파일/폴더
    heavy_items = [
        "results/profit_distribution_summary.csv",
        "results/profit_distribution/",
        "results/label_analysis/",
        "models/",
        "logs/"
    ]
    
    for item in heavy_items:
        src_path = Path(item)
        if src_path.exists():
            dst_path = backup_dir / src_path.name
            
            if src_path.is_file():
                shutil.move(str(src_path), str(dst_path))
                print(f"📁 이동: {item}")
            elif src_path.is_dir():
                shutil.move(str(src_path), str(dst_path))
                print(f"📁 이동: {item}")
    
    print(f"\n✅ 무거운 파일 정리 완료!")
    print(f"📂 백업 위치: {backup_dir}")

if __name__ == "__main__":
    cleanup_heavy_files()
