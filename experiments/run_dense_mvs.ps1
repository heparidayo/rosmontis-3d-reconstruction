param(
    [Parameter(Mandatory = $true)][string]$RunDirectory,
    [string]$Python = '.\.venv\Scripts\python.exe',
    [string]$Colmap = 'C:\Tools\COLMAP\COLMAP.bat'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$experimentPath = if ([System.IO.Path]::IsPathRooted($RunDirectory)) { $RunDirectory } else { Join-Path $projectRoot $RunDirectory }
$experimentRoot = [System.IO.Path]::GetFullPath($experimentPath)
$allowedRoot = [System.IO.Path]::GetFullPath((Join-Path $projectRoot 'runs')) + [System.IO.Path]::DirectorySeparatorChar
if (-not $experimentRoot.StartsWith($allowedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw '실험 경로는 저장소 runs 아래의 새 폴더여야 합니다.'
}
if (Test-Path -LiteralPath $experimentRoot) { throw '실험 경로가 이미 존재합니다.' }
if (-not (Test-Path -LiteralPath $Python)) { throw 'Python 실행 파일이 없습니다.' }
if (-not (Test-Path -LiteralPath $Colmap)) { throw 'COLMAP 실행 파일이 없습니다.' }
$runName = Split-Path -Leaf $experimentRoot
$diagnosticOutput = Join-Path $projectRoot ("results/diagnostics/" + $runName)
$mediaOutput = Join-Path $projectRoot ("results/media/" + $runName)
if ((Test-Path -LiteralPath $diagnosticOutput) -or (Test-Path -LiteralPath $mediaOutput)) {
    throw '같은 이름의 결과 폴더가 이미 있습니다. 새 run 이름을 사용하세요.'
}
New-Item -ItemType Directory -Path $experimentRoot | Out-Null
$workspace = Join-Path $experimentRoot 'workspace'
$stageRecords = [System.Collections.Generic.List[object]]::new()
@{
    source_model = 'runs/full360_71_20261018_02/sparse/0'
    source_images = 'runs/center_crop_71_20261018_01/images'
    image_count = 71
    undistort = @{ max_image_size = 2000; num_patch_match_src_images = 10; num_threads = 4 }
    patch_match = @{ max_image_size = 2000; gpu_index = '0'; window_radius = 7; num_iterations = 3; geom_consistency = $true; filter = $true; filter_min_num_consistent = 2; cache_size_gb = 4; num_threads = 4 }
    fusion = @{ input_type = 'geometric'; min_num_pixels = 3; use_cache = $false; num_threads = 4; cache_size_gb = 4 }
    meshing = @{ type = 'Poisson'; depth = 8; trim = 5; color = $true; num_threads = 4; preview_target_faces = 8000 }
    floor_filter = $true
    python = $Python
    colmap = $Colmap
} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $experimentRoot 'parameters.json') -Encoding utf8

function Invoke-DenseStage {
    param([string]$Name, [string]$Executable, [string[]]$Arguments)
    $logPath = Join-Path $experimentRoot ($Name + '.log')
    $started = [DateTimeOffset]::UtcNow
    $savedPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Executable @Arguments *> $logPath
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $savedPreference
    }
    $stageRecords.Add([pscustomobject]@{
        stage = $Name; executable = $Executable; arguments = $Arguments
        startedUtc = $started.ToString('o'); endedUtc = [DateTimeOffset]::UtcNow.ToString('o')
        exitCode = $exitCode; log = $logPath
    })
    ConvertTo-Json -InputObject @($stageRecords.ToArray()) -Depth 8 | Set-Content -LiteralPath (Join-Path $experimentRoot 'stages.json') -Encoding utf8
    if ($exitCode -ne 0) { throw "$Name 실패. 로그: $logPath" }
    Write-Host "$Name 완료: $logPath"
}

Invoke-DenseStage '00_audit' $Python @('src/dense_mvs_tools.py', 'audit', '--run', $experimentRoot)
foreach ($command in @('image_undistorter', 'patch_match_stereo', 'stereo_fusion', 'poisson_mesher', 'mesh_simplifier')) {
    & $Colmap $command -h *> (Join-Path $experimentRoot ($command + '_help.txt'))
    if ($LASTEXITCODE -ne 0) { throw "$command 옵션 확인 실패" }
}
Invoke-DenseStage '01_undistort' $Colmap @(
    'image_undistorter', '--image_path', (Join-Path $projectRoot 'runs/center_crop_71_20261018_01/images'),
    '--input_path', (Join-Path $projectRoot 'runs/full360_71_20261018_02/sparse/0'),
    '--output_path', $workspace, '--output_type', 'COLMAP', '--max_image_size', '2000',
    '--num_patch_match_src_images', '10', '--num_threads', '4'
)
Invoke-DenseStage '02_patch_match' $Colmap @(
    'patch_match_stereo', '--workspace_path', $workspace, '--workspace_format', 'COLMAP',
    '--PatchMatchStereo.max_image_size', '2000', '--PatchMatchStereo.gpu_index', '0',
    '--PatchMatchStereo.window_radius', '7', '--PatchMatchStereo.num_iterations', '3',
    '--PatchMatchStereo.geom_consistency', '1', '--PatchMatchStereo.filter', '1',
    '--PatchMatchStereo.filter_min_num_consistent', '2', '--PatchMatchStereo.cache_size', '4',
    '--PatchMatchStereo.num_threads', '4'
)
Invoke-DenseStage '03_fusion' $Colmap @(
    'stereo_fusion', '--workspace_path', $workspace, '--workspace_format', 'COLMAP',
    '--input_type', 'geometric', '--output_path', (Join-Path $workspace 'fused.ply'),
    '--StereoFusion.min_num_pixels', '3', '--StereoFusion.num_threads', '4', '--StereoFusion.cache_size', '4', '--StereoFusion.use_cache', '0'
)
Invoke-DenseStage '04_object_roi' $Python @('src/dense_mvs_tools.py', 'crop', '--run', $experimentRoot)
Invoke-DenseStage '04b_floor_filter' $Python @('src/dense_mvs_tools.py', 'floor-filter', '--run', $experimentRoot)
Invoke-DenseStage '05_depth_report' $Python @('src/dense_mvs_tools.py', 'depth-report', '--run', $experimentRoot, '--output', $diagnosticOutput)
$rawMesh = Join-Path $experimentRoot 'object_poisson.ply'
$cleanMesh = Join-Path $experimentRoot 'object_poisson_clean.ply'
$previewMesh = Join-Path $experimentRoot 'object_mesh_8k.ply'
$coloredMesh = Join-Path $experimentRoot 'object_mesh_8k_clean_color.ply'
Invoke-DenseStage '06_poisson' $Colmap @(
    'poisson_mesher', '--input_path', (Join-Path $experimentRoot 'object_above_floor.ply'),
    '--output_path', $rawMesh, '--PoissonMeshing.depth', '8', '--PoissonMeshing.trim', '5',
    '--PoissonMeshing.color', '1', '--PoissonMeshing.num_threads', '4'
)
Invoke-DenseStage '07_clean_mesh' $Python @('src/dense_mvs_tools.py', 'clean-mesh', '--run', $experimentRoot, '--mesh', $rawMesh, '--output', $cleanMesh)
$countCode = "import sys; sys.path.insert(0,'src'); from dense_mvs_tools import read_ply; print(len(read_ply(sys.argv[1])[1]))"
$faceCount = [int](& $Python -c $countCode $cleanMesh)
if ($LASTEXITCODE -ne 0 -or $faceCount -lt 1) { throw '메시 면 수 확인 실패' }
$targetRatio = [Math]::Min(1.0, 8000.0 / $faceCount).ToString('R', [Globalization.CultureInfo]::InvariantCulture)
Invoke-DenseStage '08_simplify' $Colmap @(
    'mesh_simplifier', '--input_path', $cleanMesh, '--output_path', $previewMesh,
    '--MeshSimplification.target_face_ratio', $targetRatio,
    '--MeshSimplification.interpolate_colors', '1', '--MeshSimplification.num_threads', '4'
)
Invoke-DenseStage '09_color_cleanup' $Python @('src/dense_mvs_tools.py', 'clean-mesh', '--run', $experimentRoot, '--mesh', $previewMesh, '--color-source', $cleanMesh, '--output', $coloredMesh)
Invoke-DenseStage '10_render' $Python @('src/dense_mvs_tools.py', 'render', '--run', $experimentRoot, '--mesh', $coloredMesh, '--cloud', (Join-Path $experimentRoot 'object_above_floor.ply'), '--output', $mediaOutput)
Invoke-DenseStage '11_mesh_projection' $Python @('src/dense_mvs_tools.py', 'mesh-projection', '--run', $experimentRoot, '--mesh', $coloredMesh, '--output', $diagnosticOutput)
Invoke-DenseStage '12_verify_sources' $Python @('src/dense_mvs_tools.py', 'verify', '--run', $experimentRoot)
Write-Host "Dense 점군·메시·GIF 생성 완료. 직접 시각 검증할 경로: $mediaOutput"
