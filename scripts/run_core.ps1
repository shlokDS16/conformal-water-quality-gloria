# Core run matrix launcher (research/EXPERIMENT_PLAN.md s8). Revised after AUDIT_3 (research/FIX_AUDIT3_LOG.md).
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_core.ps1 -DryRun      # print every command and counts, run nothing
#   powershell -ExecutionPolicy Bypass -File scripts\run_core.ps1              # start both streams in background
#   powershell -ExecutionPolicy Bypass -File scripts\run_core.ps1 -Stream gpu  # run one stream in this window
#
# Two streams run in parallel: "gpu" (every job that trains MDN; GPU vmap + 12 CPU workers x 1 LightGBM thread)
# and "cpu" (LightGBM-only jobs; 10 workers x 1 thread). Each job is resumable: finished splits are skipped,
# so a stream can be restarted after a crash or reboot. Logs: logs\core_<stream>_<timestamp>.log plus one log per job.
# Queue order: core cells first in both streams (gpu: hyperspectral core with the confirmatory waterbody and random
# cells first; cpu: calibration budget, prereg s6), then sensor cells, then sensitivities.
# Tags: fresh directories only (core_v1, budget_v1, sens_v1, noise_v1 are superseded by AUDIT_3 fixes).
# Noise sweep (Amendment 3): levels mult:add, default "0.27:0.0004","0.39:0.0005","0.70:0.001"; -NoNoise disables it.
# 5 km grouping sensitivity (Amendment 3 item 3): protocol waterbody_5km, split files data/processed/splits5km_<t>.parquet.

param(
    [ValidateSet("", "gpu", "cpu")] [string] $Stream = "",
    [switch] $DryRun,
    [string[]] $NoiseLevels = @("0.27:0.0004", "0.39:0.0005", "0.70:0.001"),
    [switch] $NoNoise,
    [string] $Tag = "core_v2",
    [string] $BudgetTag = "budget_v2",
    [string] $SensTag = "sens_v2",
    [string] $NoiseTag = "noise_v2",
    [int] $GpuJobs = 12,
    [int] $CpuJobs = 10
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
New-Item -ItemType Directory -Force "logs" | Out-Null
$env:PYTHONHASHSEED = "0"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:OMP_NUM_THREADS = "1"

$Targets = @("Chla", "TSS", "aCDOM440", "Secchi_depth")
$NonMdn = "const,ridge,emp,lgbm"
$Cheap = "scp_pool,scp_gsub,cqr_pool,cqr_gsub"
$Gsub = "scp_gsub,cqr_gsub"
$GpuTail = "--n-jobs $GpuJobs --lgbm-threads 1 --mdn-chunk 1400"   # --mdn-chunk fixed within a tag (AUDIT_3 A3-9)
$CpuTail = "--n-jobs $CpuJobs --lgbm-threads 1"
$Levels = @()
if (-not $NoNoise) { $Levels = @(($NoiseLevels -join ",").Split(",") | Where-Object { $_ }) }

function Exp([string]$a) { "python -m src.run_experiment $a" }
function J([string]$block, [string]$cmd) { [pscustomobject]@{ Block = $block; Cmd = $cmd } }

function Get-GpuJobs {
    $j = @()
    # A: hyperspectral core, all models and methods; confirmatory cells (waterbody, random) first
    foreach ($p in @("waterbody", "random", "contributor_ds", "region", "region_na")) { foreach ($t in $Targets) {
        $j += J "A_hyp_core" (Exp "--target $t --sensor hyp --protocol $p --tag $Tag $GpuTail") } }
    # B1: MSI, MDN on random, waterbody, region, region_na, contributor_ds seeds 0-4
    foreach ($t in $Targets) { foreach ($p in @("waterbody", "random", "region", "region_na")) {
        $j += J "B1_msi_mdn" (Exp "--target $t --sensor msi --protocol $p --tag $Tag $GpuTail") }
        $j += J "B1_msi_mdn" (Exp "--target $t --sensor msi --protocol contributor_ds --seeds 0-4 --tag $Tag $GpuTail") }
    # B2: OLCI, MDN on random and waterbody
    foreach ($t in $Targets) { foreach ($p in @("waterbody", "random")) {
        $j += J "B2_olci_mdn" (Exp "--target $t --sensor olci --protocol $p --tag $Tag $GpuTail") } }
    # F: noise sweep (prereg s7, Amendment 3): MSI and OLCI, LightGBM and MDN, waterbody seeds 0-4
    foreach ($lv in $Levels) {
        $m, $a = $lv.Split(":")
        foreach ($t in $Targets) { foreach ($s in @("msi", "olci")) {
            $j += J "F_noise" (Exp "--target $t --sensor $s --protocol waterbody --seeds 0-4 --models lgbm,mdn --noise-mult $m --noise-add $a --tag $NoiseTag $GpuTail") } } }
    return $j
}

function Get-CpuJobs {
    $j = @()
    # E: calibration budget (prereg s6, core contribution)
    foreach ($t in $Targets) { $j += J "E_budget" "python -m src.run_budget --mode group --target $t --seeds 0-19 --draws 20 --tag $BudgetTag --n-jobs $CpuJobs" }
    foreach ($t in @("Chla", "TSS")) { foreach ($o in @("earliest", "random")) {
        $j += J "E_budget" "python -m src.run_budget --mode local --target $t --seeds 0-19 --order $o --tag $BudgetTag --n-jobs $CpuJobs" } }
    # B1/B2 non-MDN remainder (sensor cells)
    foreach ($t in $Targets) {
        $j += J "B_sensor_nonmdn" (Exp "--target $t --sensor msi --protocol contributor_ds --seeds 5-9 --models $NonMdn --tag $Tag $CpuTail")
        foreach ($p in @("contributor_ds", "region", "region_na")) {
            $j += J "B_sensor_nonmdn" (Exp "--target $t --sensor olci --protocol $p --models $NonMdn --tag $Tag $CpuTail") } }
    # C: contrib_group sensitivity (hyp, non-MDN; A1.2)
    foreach ($t in $Targets) { $j += J "C_contrib_group" (Exp "--target $t --sensor hyp --protocol contributor --models $NonMdn --tag $Tag $CpuTail") }
    # D: population sensitivities, LightGBM, waterbody seeds 0-9 (prereg s7; A2.2 exploratory exclude_nonpositive)
    foreach ($t in $Targets) { foreach ($pop in @("strict_qc", "depth_ge3", "exclude_nonpositive")) {
        $j += J "D_population" (Exp "--target $t --sensor hyp --protocol waterbody --seeds 0-9 --population $pop --models lgbm --methods $Cheap --tag $SensTag $CpuTail") } }
    $j += J "D_population" (Exp "--target Chla --sensor hyp --protocol waterbody --seeds 0-9 --population chla_hplc --models lgbm --methods $Cheap --tag $SensTag $CpuTail")
    # D2: Secchi rows with Depth (A2.8; AUDIT_3 A3-4)
    $j += J "D2_secchi_has_depth" (Exp "--target Secchi_depth --sensor hyp --protocol waterbody --seeds 0-9 --population secchi_has_depth --models lgbm --methods $Gsub --tag $SensTag $CpuTail")
    # D3: strict band sets MSI B1-B4, OLCI Oa2-Oa10 (A1.1; AUDIT_3 A3-4)
    foreach ($t in $Targets) { foreach ($s in @("msi_strict", "olci_strict")) {
        $j += J "D3_strict_bands" (Exp "--target $t --sensor $s --protocol waterbody --seeds 0-9 --models lgbm --methods $Gsub --tag $SensTag $CpuTail") } }
    # G: 5 km grouping sensitivity (prereg s7; Amendment 3 item 3)
    foreach ($t in $Targets) {
        $j += J "G_5km" (Exp "--target $t --sensor hyp --protocol waterbody_5km --seeds 0-9 --models lgbm --methods $Gsub --tag $SensTag $CpuTail") }
    return $j
}

function Invoke-Stream([string]$name, $jobs) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $master = "logs\core_${name}_$stamp.log"
    $i = 0
    foreach ($job in $jobs) {
        $i++
        $cmd = $job.Cmd
        $jobLog = "logs\core_${name}_$stamp`_job$('{0:D3}' -f $i).log"
        "$(Get-Date -Format s) START [$i/$($jobs.Count)] [$($job.Block)] $cmd" | Out-File -Append -Encoding utf8 $master
        $t0 = Get-Date
        cmd /c "$cmd > `"$jobLog`" 2>&1"
        $code = $LASTEXITCODE
        $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
        "$(Get-Date -Format s) END   [$i/$($jobs.Count)] exit=$code minutes=$mins log=$jobLog" | Out-File -Append -Encoding utf8 $master
        if ($code -ne 0) { "$(Get-Date -Format s) FAILED job $i; continuing with next job (rerun the stream to retry)" | Out-File -Append -Encoding utf8 $master }
    }
    "$(Get-Date -Format s) STREAM DONE" | Out-File -Append -Encoding utf8 $master
}

$gpu = @(Get-GpuJobs)
$cpu = @(Get-CpuJobs)
if ($DryRun) {
    foreach ($pair in @(@("GPU", $gpu), @("CPU", $cpu))) {
        "$($pair[0]) stream: $($pair[1].Count) jobs"
        $i = 0
        foreach ($job in $pair[1]) { $i++; "  [{0:D3}] [{1}] {2}" -f $i, $job.Block, $job.Cmd }
    }
    ""
    "Summary (tags: core $Tag, budget $BudgetTag, sensitivity $SensTag, noise $NoiseTag; noise levels: $($Levels -join ' '))"
    foreach ($pair in @(@("GPU", $gpu), @("CPU", $cpu))) {
        foreach ($g in ($pair[1] | Group-Object Block)) { "  {0} {1,-22} {2,3} jobs" -f $pair[0], $g.Name, $g.Count }
    }
    "  TOTAL: GPU $($gpu.Count) + CPU $($cpu.Count) = $($gpu.Count + $cpu.Count) jobs"
    exit 0
}
if ($Stream -eq "gpu") { Invoke-Stream "gpu" $gpu; exit 0 }
if ($Stream -eq "cpu") { Invoke-Stream "cpu" $cpu; exit 0 }

# No stream given: start both streams as hidden background processes and return.
$self = $MyInvocation.MyCommand.Path
$common = "-NoProfile -ExecutionPolicy Bypass -File `"$self`" -Tag $Tag -BudgetTag $BudgetTag -SensTag $SensTag -NoiseTag $NoiseTag -GpuJobs $GpuJobs -CpuJobs $CpuJobs"
if ($NoNoise) { $common += " -NoNoise" } else { $common += " -NoiseLevels " + ($Levels -join ",") }
Start-Process powershell -WindowStyle Hidden -ArgumentList "$common -Stream gpu"
Start-Process powershell -WindowStyle Hidden -ArgumentList "$common -Stream cpu"
"Started gpu and cpu streams; follow logs\core_gpu_*.log and logs\core_cpu_*.log"
