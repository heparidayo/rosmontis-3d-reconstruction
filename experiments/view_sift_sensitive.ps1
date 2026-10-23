param(
    [ValidateSet('0', '1')]
    [string]$Model = '1'
)

$project = Split-Path -Parent $PSScriptRoot
$experiment = Join-Path $project 'runs\sift_peak_003_71_20261018_01\peak_003'
$database = Join-Path $experiment 'database.db'
$images = Join-Path $project 'runs\center_crop_71_20261018_01\images'
$sparse = Join-Path $experiment "sparse\$Model"
foreach ($path in @($database, $images, $sparse)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "경로가 없습니다: $path" }
}

# 기본값은 새로 복원된 후면 11장 모델. -Model 0은 정면 쪽 38장 모델.
& 'C:\Tools\COLMAP\COLMAP.bat' gui --database_path $database --image_path $images --import_path $sparse
if ($LASTEXITCODE -ne 0) { throw "COLMAP GUI 종료 코드: $LASTEXITCODE" }
