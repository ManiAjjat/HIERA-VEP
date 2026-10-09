param([string]$Python = "python")
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
function Require-Success([string]$Step) {
    if ($LASTEXITCODE -ne 0) { throw "$Step failed (exit $LASTEXITCODE). Publication stopped." }
}
& $Python -c "import sys; assert sys.version_info[:2] == (3,11)"
Require-Success "Python version check"
& $Python -m pytest -q
Require-Success "Tests"
& $Python -m compileall -q src HIERA_VEP_COMPLETE_CODE.py
Require-Success "Compilation"
& $Python src/hiera_vep_score.py --bundle . --input examples/example_scoring_input.tsv --output examples/example_predictions.tsv
Require-Success "Example inference"
& $Python -c "from pathlib import Path; import hashlib; r=Path('.'); lines=(r/'CHECKSUMS.sha256').read_text().splitlines(); assert all(hashlib.sha256((r/l.split('  ',1)[1]).read_bytes()).hexdigest()==l.split('  ',1)[0] for l in lines)"
Require-Success "Checksums"
gh auth status
Require-Success "GitHub authentication"
$Login = gh api user --jq .login
Require-Success "GitHub ownership check"
if ($Login -ne "ManiAjjat") { throw "Authenticated account must be ManiAjjat. Publication stopped." }
$Existing = gh repo view ManiAjjat/HIERA-VEP --json name 2>&1
if ($LASTEXITCODE -eq 0) { throw "ManiAjjat/HIERA-VEP already exists. Explicit conflict resolution is required." }
if (($Existing | Out-String) -notmatch "Could not resolve to a Repository") { throw "Unable to establish repository absence. Publication stopped." }
git status --porcelain
Require-Success "Git status"
$Dirty = git status --porcelain
if ($Dirty) { throw "Commit the reviewed clean release before publishing." }
gh repo create ManiAjjat/HIERA-VEP --public --source . --remote origin --push --description "Interpretable context-gated regulatory variant-effect prediction from sequence and genomic annotations."
Require-Success "Repository creation and push"
gh repo view ManiAjjat/HIERA-VEP --json url,visibility,defaultBranchRef
Require-Success "Remote verification"
gh run list --repo ManiAjjat/HIERA-VEP --limit 5
Require-Success "Workflow visibility"
Write-Host "Main branch published. Verify the remote files and successful Actions run before creating v1.0.0. See docs/GITHUB_PUBLISHING.md."
