# run_generate.ps1
# ----------------
# Windows equivalent of run_generate.sh: every archive of the full zone with
# `xphd generate`, several at once, skipping any that exist and are complete.
#
# Before running: generate the Gamma archive by hand and make sure
#     xphd check-archive <Out>\GI_ExcPh_Q0001.npz
# prints READY. The script refuses to start otherwise.
#
# -Parallel needs PowerShell 7 (pwsh). Windows PowerShell 5.1 does not have it
# (check $PSVersionTable.PSVersion); on 5.1 use -Throttle 1 and it runs serially.
# -Throttle is bounded by MEMORY: each process loads the whole el-ph database.
#
# Examples:
#   GaN : .\run_generate.ps1 -N 24 -NExc 15 -Save ..\ELPH\SAVE -Elph ..\ELPH\ndb.elph -Bse ..\BSE_EXCPH\output
#   hBN : .\run_generate.ps1 -N 24 -NExc 15 -Save ..\phonons\SAVE -Elph ..\phonons\ndb.elph -Bse ..\BSE\output_all
#   WSe2: .\run_generate.ps1 -N 12 -NExc 10 -Save ..\LELPH\SAVE -Elph <path>\ndb.elph -Bse ..\BSE\output

param(
    [int]$N = 24,
    [int]$Throttle = 4,
    [int]$NExc = 15,
    [string]$Out = "archives",
    [string]$Save = "..\ELPH\SAVE",
    [string]$Elph = "..\ELPH\ndb.elph",
    [string]$Bse = "..\BSE_EXCPH\output",
    [string]$Dmats = "Dmats.npy"
)

$NQ = $N * $N
New-Item -ItemType Directory -Force -Path $Out, logs | Out-Null

$g0 = Join-Path $Out "GI_ExcPh_Q0001.npz"
$ready = (Test-Path $g0) -and ((xphd check-archive $g0) -match "READY: launch the rest")
if (-not $ready) {
    Write-Host "CRITICAL: $g0 missing or not READY. Generate it by hand first:"
    Write-Host "  xphd generate 0 $Out --nexc $NExc --savepath $Save --ndb-elph $Elph --bse-dir $Bse --dmats $Dmats --mesh $N $N"
    Write-Host "  xphd check-archive $g0"
    exit 1
}

$check = @"
import numpy as np, sys
d = np.load(sys.argv[1])
need = {'G_grid','Ge_grid','Gh_grid','g2_grid','mesh'}
sys.exit(0 if need <= set(d.files) else 1)
"@
Set-Content -Path "_complete.py" -Value $check

Write-Host "generating $NQ archives ($N x $N), $Throttle at a time, $(Get-Date)"

1..($NQ - 1) | ForEach-Object -ThrottleLimit $Throttle -Parallel {
    $iQ = $_
    $f = "{0}\GI_ExcPh_Q{1:D4}.npz" -f $using:Out, ($iQ + 1)
    if ((Test-Path $f) -and ((Get-Item $f).Length -gt 0)) {
        python _complete.py $f 2>$null
        if ($LASTEXITCODE -eq 0) { Write-Host "skip $iQ (complete)"; return }
    }
    $log = "logs\gen_{0:D4}.log" -f ($iQ + 1)
    xphd generate $iQ $using:Out --nexc $using:NExc --savepath $using:Save `
        --ndb-elph $using:Elph --bse-dir $using:Bse --dmats $using:Dmats `
        --mesh $using:N $using:N *> $log
    if ($LASTEXITCODE -eq 0) { Write-Host "done $iQ" }
    else { Write-Host "FAILED $iQ -- see $log" }
}

Write-Host "finished $(Get-Date)"
xphd verify-archives --dir $Out --n $NQ
