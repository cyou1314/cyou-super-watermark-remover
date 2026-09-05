[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [Parameter(Mandatory = $true)]
    [ValidateRange(0, 100000)]
    [int]$X,

    [Parameter(Mandatory = $true)]
    [ValidateRange(0, 100000)]
    [int]$Y,

    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 100000)]
    [int]$Width,

    [Parameter(Mandatory = $true)]
    [ValidateRange(1, 100000)]
    [int]$Height
)

$ErrorActionPreference = 'Stop'

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    throw 'ffmpeg was not found on PATH.'
}

if (-not (Get-Command ffprobe -ErrorAction SilentlyContinue)) {
    throw 'ffprobe was not found on PATH.'
}

$resolvedInput = (Resolve-Path -LiteralPath $InputPath).Path
$outputFullPath = [System.IO.Path]::GetFullPath($OutputPath)

if ($resolvedInput -eq $outputFullPath) {
    throw 'OutputPath must be different from InputPath.'
}

if (Test-Path -LiteralPath $outputFullPath) {
    throw "Output already exists: $outputFullPath"
}

$outputDirectory = Split-Path -Parent $outputFullPath
if (-not (Test-Path -LiteralPath $outputDirectory -PathType Container)) {
    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
}

$videoInfo = ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of json $resolvedInput | ConvertFrom-Json
if (-not $videoInfo.streams -or $videoInfo.streams.Count -ne 1) {
    throw 'Expected exactly one video stream.'
}

$videoWidth = [int]$videoInfo.streams[0].width
$videoHeight = [int]$videoInfo.streams[0].height
if (($X + $Width) -gt $videoWidth -or ($Y + $Height) -gt $videoHeight) {
    throw "Mask exceeds video bounds ${videoWidth}x${videoHeight}."
}

$filter = "delogo=x=${X}:y=${Y}:w=${Width}:h=${Height}:show=0"
$arguments = @(
    '-hide_banner',
    '-n',
    '-benchmark',
    '-i', $resolvedInput,
    '-map', '0:v:0',
    '-map', '0:a?',
    '-vf', $filter,
    '-c:v', 'libx264',
    '-crf', '18',
    '-preset', 'fast',
    '-pix_fmt', 'yuv420p',
    '-c:a', 'copy',
    '-movflags', '+faststart',
    $outputFullPath
)

& ffmpeg @arguments
if ($LASTEXITCODE -ne 0) {
    throw "ffmpeg failed with exit code $LASTEXITCODE."
}

ffprobe -v error -show_format -show_streams -of json $outputFullPath
if ($LASTEXITCODE -ne 0) {
    throw "ffprobe verification failed with exit code $LASTEXITCODE."
}
