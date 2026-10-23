# 중앙 크롭 71장 실험의 35카메라 Sparse 모델을 확인한다.
$ErrorActionPreference = 'Stop'
$resultPath = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\runs\center_crop_71_20261018_01'))
$databasePath = Join-Path $resultPath 'database.db'
$imagePath = Join-Path $resultPath 'images'
$modelPath = Join-Path $resultPath 'sparse\0'
if (-not (Test-Path -LiteralPath $databasePath -PathType Leaf)) { throw "DB 없음: $databasePath" }
if (-not (Test-Path -LiteralPath (Join-Path $modelPath 'images.bin') -PathType Leaf)) { throw "모델 없음: $modelPath" }
& 'C:\Tools\COLMAP\COLMAP.bat' gui --database_path $databasePath --image_path $imagePath --import_path $modelPath
