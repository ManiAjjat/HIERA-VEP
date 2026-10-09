# GitHub publication

The target is the public repository `ManiAjjat/HIERA-VEP`. Publication requires authentication as its owner. Use `gh auth login --hostname github.com --git-protocol https --web`; complete GitHub's official browser/device flow. Never put a password or token in an issue, file, command argument, or chat.

The repository was prepared locally with original scientific scripts preserved. Before publication, review `docs/RELEASE_AUDIT.md`, tests, checksums and the file list. `PUBLISH_TO_GITHUB.ps1` verifies tests, inference, checksums, ownership and repository absence before creating and pushing the target. It stops on an existing repository and never force-pushes. Run from a committed checkout using Python 3.11:

```powershell
./PUBLISH_TO_GITHUB.ps1 -Python python
```

Verify the remote README, main figures, model card, assets, examples, sources, license, citation and workflow. Confirm the Actions inference tests pass before release. Only then:

```powershell
git tag -a v1.0.0 -m "HIERA-VEP v1.0.0"
git push origin v1.0.0
gh release create v1.0.0 --repo ManiAjjat/HIERA-VEP --title "HIERA-VEP v1.0.0" --notes-file RELEASE_NOTES.md
```

Create a clean ZIP from the tracked commit with `git archive --format=zip --output ../HIERA-VEP-v1.0.0.zip HEAD`; attach it with `gh release upload v1.0.0 ../HIERA-VEP-v1.0.0.zip --repo ManiAjjat/HIERA-VEP`. Do not zip the working directory, environments, or `.git`. Final publication status is reported separately; these instructions alone do not establish that a repository or release exists.
