$project = Split-Path -Parent $PSScriptRoot
$run = Join-Path $project 'runs\full360_71_20261018_02'
$images = Join-Path $project 'runs\center_crop_71_20261018_01\images'
$database = Join-Path $run 'database.db'
$model = Join-Path $run 'sparse\0'
foreach ($path in @($database, $images, $model)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "경로가 없습니다: $path" }
}
& 'C:\Tools\COLMAP\COLMAP.bat' gui --database_path $database --image_path $images --import_path $model
if ($LASTEXITCODE -ne 0) { throw "COLMAP GUI 종료 코드: $LASTEXITCODE" }
