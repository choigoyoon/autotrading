### 1단계: 최종 산출물 확인 및 검수

**종합 분석 대시보드 열람 (가장 중요!)**
```bash
# Windows에서 직접 파일 열기
start results/comprehensive_dashboard.html

# 또는 브라우저에서 직접 열기 (로컬 서버 실행)
python -m http.server 8000  # 로컬 서버 실행 후 localhost:8000에서 확인
```

**전체 산출물 구조 확인**
```bash
# Windows에서는 tree가 기본 명령어가 아닐 수 있으므로, dir 사용
dir /s results

# 각 폴더 내용 확인
dir results\v2_analysis
dir results\v2_optimization
dir results\simulation
```

### 2단계: 프로젝트 백업 및 아카이브

**전체 결과물 백업 (타임스탬프 포함)**
```powershell
# PowerShell을 사용한 압축
$timestamp = Get-Date -Format "yyyyMMdd_HHmm"
Compress-Archive -Path results, tools, configs -DestinationPath "Enhanced_System_Results_$timestamp.zip"
```

**핵심 파일만 별도 백업**
```bash
mkdir backup\core_files
copy results\comprehensive_dashboard.html backup\core_files\
copy tools\create_hybrid_system.py backup\core_files\
copy configs\hybrid_config.yaml backup\core_files\
copy results\simulation\performance_comparison.csv backup\core_files\
```

### 3단계: 실전 적용 모듈 추출

**실전 매매용 폴더 생성 및 모듈 복사**
```bash
mkdir production_trading
copy tools\create_hybrid_system.py production_trading\hybrid_trading_engine.py
copy configs\hybrid_config.yaml production_trading\
copy tools\simulate_realistic_performance.py production_trading\performance_monitor.py
```

**실전 설정 파일 커스터마이징 준비**
```powershell
# PowerShell을 사용하여 파일 생성
Set-Content -Path production_trading\live_config.yaml -Value "# 실전 매매 설정 - 리스크 관리 강화"
```

### 4단계: 자동화 시스템 구축

**월간 자동 최적화 스크립트 생성 (PowerShell 스크립트)**
```powershell
@'
#!/usr/bin/env powershell
# 월간 시스템 최적화 자동 실행
python tools/optimize_v2_thresholds.py
python tools/simulate_realistic_performance.py  
python tools/comprehensive_final_analysis.py
"월간 최적화 완료: $(Get-Date)" | Out-File -Append -FilePath optimization_log.txt
'@ | Set-Content -Path monthly_optimization.ps1
```

**Windows 작업 스케줄러 등록 예시 (매월 1일 실행)**
- 작업 스케줄러를 열고 새 작업을 만듭니다.
- 트리거: '매월', '1일'로 설정합니다.
- 동작: '프로그램 시작', 프로그램/스크립트 란에 `powershell.exe`를 입력하고 인수 추가 란에 `-File "C:\path\to\your\project\monthly_optimization.ps1"` 와 같이 스크립트 전체 경로를 입력합니다. 