# 기존 DB/이미지/Sparse를 보존한 Mapper-only A/B. 이 스크립트 실행 전 보고서를 읽는다.
[CmdletBinding()]
param(
    [string]$SourceRun = (Join-Path $PSScriptRoot '..\runs\multi_angle_48'),
    [string]$OutputRoot = (Join-Path $PSScriptRoot ('..\runs\radial_ab_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))),
    [string]$Colmap = 'C:\Tools\COLMAP\COLMAP.bat',
    [string]$Python = (Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe')
)
$ErrorActionPreference = 'Stop'
$sourcePath = (Resolve-Path -LiteralPath $SourceRun).Path
$destinationPath = [IO.Path]::GetFullPath($OutputRoot)
if (Test-Path -LiteralPath $destinationPath) {
    throw "새 출력 경로가 필요합니다: $destinationPath"
}
$sourcePrefix = $sourcePath.TrimEnd('\') + '\'
if ($destinationPath.StartsWith($sourcePrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw '원본 run 내부에는 출력할 수 없습니다.'
}
if (-not (Test-Path -LiteralPath $Colmap -PathType Leaf)) { throw "COLMAP 없음: $Colmap" }
& $Python -c 'import sqlite3' 2>$null
if ($LASTEXITCODE -ne 0) { throw 'Python 실행 실패. -Python에 작동하는 Python 경로를 지정하세요.' }
# 설치된 CLI에서 실제 옵션을 다시 확인한다.
$helpText = (& $Colmap mapper -h 2>&1 | Out-String)
if ($helpText -notmatch '--Mapper.ba_refine_extra_params' -or $helpText -notmatch '--Mapper.random_seed') {
    throw '필요한 Mapper 옵션을 확인할 수 없습니다.'
}
$null = New-Item -ItemType Directory -Path $destinationPath
$helpText | Set-Content -LiteralPath (Join-Path $destinationPath 'mapper_help.txt') -Encoding UTF8
$sourceDb = Join-Path $sourcePath 'database.db'
$beforeHash = (Get-FileHash -LiteralPath $sourceDb -Algorithm SHA256).Hash
$backupCode = @'
import pathlib, sqlite3, sys
source, target = map(pathlib.Path, sys.argv[1:])
if target.exists():
    raise FileExistsError(target)
with sqlite3.connect(source.resolve().as_uri()+'?mode=ro', uri=True) as src:
    with sqlite3.connect(target) as dst:
        src.backup(dst)
'@
foreach ($arm in @('baseline', 'fixed_radial')) {
    $runPath = Join-Path $destinationPath $arm
    $sparsePath = Join-Path $runPath 'sparse'
    $null = New-Item -ItemType Directory -Path $sparsePath
    $dbPath = Join-Path $runPath 'database.db'
    & $Python -c $backupCode $sourceDb $dbPath
    if ($LASTEXITCODE -ne 0) { throw "DB 복사 실패: $arm" }
    $refine = if ($arm -eq 'baseline') { '1' } else { '0' }
    $arguments = @('mapper', '--database_path', $dbPath,
                   '--image_path', (Join-Path $sourcePath 'images'),
                   '--output_path', $sparsePath,
                   '--Mapper.random_seed', '0', '--default_random_seed', '0',
                   '--Mapper.ba_refine_extra_params', $refine,
                   '--log_level', '2', '--log_target', 'stderr', '--log_color', '0')
    [ordered]@{source_run=$sourcePath; source_db_sha256=$beforeHash;
        change='Mapper.ba_refine_extra_params'; value=$refine; arguments=$arguments;
        image_count=(Get-ChildItem -LiteralPath (Join-Path $sourcePath 'images') -File -Filter '*.JPG').Count
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $runPath 'experiment.json') -Encoding UTF8
    # CMD를 호출하지 않고 PowerShell의 동일 셸에서 실행·로그를 기록한다.
    # COLMAP의 INFO도 stderr이므로 Windows PowerShell에서는 계속 처리한다.
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { & $Colmap @arguments 2>&1 | Tee-Object -FilePath (Join-Path $runPath 'mapper.log') }
    finally { $ErrorActionPreference = $previousPreference }
    if ($LASTEXITCODE -ne 0) { throw "Mapper 실패: $arm" }
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        Get-ChildItem -LiteralPath $sparsePath -Directory | ForEach-Object {
            & $Colmap model_analyzer --path $_.FullName 2>&1 |
                Out-File -LiteralPath (Join-Path $runPath ('model_' + $_.Name + '_analyzer.txt')) -Encoding UTF8
            if ($LASTEXITCODE -ne 0) { throw "model_analyzer 실패: $($_.Name)" }
        }
    } finally { $ErrorActionPreference = $previousPreference }
}
$afterHash = (Get-FileHash -LiteralPath $sourceDb -Algorithm SHA256).Hash
if ($beforeHash -ne $afterHash) { throw '원본 DB가 분석 중 변경됐습니다. 동시 실행 여부를 확인하세요.' }
Write-Host "완료: $destinationPath"
Write-Host '모든 Sparse 모델의 등록 이미지 합집합, 경계 이미지 추가 여부, k, track/error를 비교하세요.'
