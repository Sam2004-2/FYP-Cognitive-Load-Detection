#!/usr/bin/env pwsh
# Parallel feature extraction using multiple PowerShell terminals
# Run this script to process AVCAffe dataset 4x faster

$inputDir = "E:\FYP\Dataset\AVCAffe\codes\downloader\data\videos\per_participant_per_task"
$baseOutput = "E:\FYP\FYP-Cognitive-Load-Detection\Machine Learning\data\processed"
$scriptDir = "E:\FYP\FYP-Cognitive-Load-Detection\Machine Learning"
$python = "E:\FYP\.venv\Scripts\python.exe"

# 108 participants total, split into 4 chunks of ~27 each
$chunks = @(
    @{ start = 0;  count = 27; output = "avcaffe_part1.csv" },
    @{ start = 27; count = 27; output = "avcaffe_part2.csv" },
    @{ start = 54; count = 27; output = "avcaffe_part3.csv" },
    @{ start = 81; count = 27; output = "avcaffe_part4.csv" }
)

Write-Host "Starting 4 parallel extraction processes..." -ForegroundColor Green
Write-Host ""

foreach ($chunk in $chunks) {
    $cmd = "cd '$scriptDir'; `$env:PYTHONPATH = '.'; & '$python' scripts/extract_video_features.py --input_dir '$inputDir' --output '$baseOutput/$($chunk.output)' --workers 1 --start_participant $($chunk.start) --max_participants $($chunk.count)"
    
    Write-Host "Starting: Participants $($chunk.start + 1) to $($chunk.start + $chunk.count)" -ForegroundColor Cyan
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $cmd
}

Write-Host ""
Write-Host "All 4 processes started in separate windows." -ForegroundColor Green
Write-Host "Monitor progress in each window." -ForegroundColor Yellow
Write-Host ""
Write-Host "After all complete, merge with:" -ForegroundColor Yellow
Write-Host "  Get-Content $baseOutput\avcaffe_part*.csv | Select-Object -Skip 1 | Set-Content $baseOutput\avcaffe_features_all.csv"
