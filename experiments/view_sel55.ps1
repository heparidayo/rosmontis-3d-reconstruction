# 71장 입력으로 복원한 32카메라 Sparse 결과를 COLMAP GUI로 연다.
$ErrorActionPreference = 'Stop'
$resultPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\runs\three_elevations_sel55_20261018_01'))
$databasePath = Join-Path $resultPath 'database.db'
$imagePath = Join-Path $resultPath 'images'
$modelPath = Join-Path $resultPath 'sparse\0'
if (-not (Test-Path -LiteralPath $databasePath -PathType Leaf)) { throw "DB 없음: $databasePath" }
if (-not (Test-Path -LiteralPath (Join-Path $modelPath 'images.bin') -PathType Leaf)) { throw "모델 없음: $modelPath" }
& 'C:\Tools\COLMAP\COLMAP.bat' gui --database_path $databasePath --image_path $imagePath --import_path $modelPath
