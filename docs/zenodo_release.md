# Zenodo release flow

This project mints a citable DOI on Zenodo for every GitHub release.

- **Concept DOI** — `[TBD on first auto-release]` — always resolves to the latest version.
- **Version DOIs** — `[TBD on first auto-release]` — pin to a specific release.

The previous manual deposit `10.5281/zenodo.20640736` (v0.3.1, June 2026) is
the v1 short-paper artifact and remains citable; the auto-linked flow below
covers all subsequent versioned releases (v2.0.0 onwards).

---

## One-time setup (done once per repo, by the owner)

1. Sign into Zenodo with your GitHub account: <https://zenodo.org/account/settings/github/>.
2. Locate `ShahnawazKakarh/speech-emotion-recognition-transfer-learning` in the repository list.
3. Toggle the switch to **ON**.

From this point onward every GitHub release on this repo will auto-deposit
to Zenodo and mint a versioned DOI; the first such release also mints a
Concept DOI that always points at the latest version.

---

## Per-release flow

Each release follows the same six steps. Aim for one release per published paper or major scope bump.

### 1. Decide the version number

Use semantic versioning with the following project conventions:

| Bump | When |
|---|---|
| **Major** `vX.0.0` | New paper / new principal finding / breaking change to the public API or data layout |
| **Minor** `vX.Y.0` | New experiment, new dataset, new analysis script — additive |
| **Patch** `vX.Y.Z` | Bug fixes, doc updates, dependency bumps |

### 2. Update `CITATION.cff`

Bump `version` and `date-released`. If the release is tied to a specific paper
also update the `preferred-citation` block (title, DOI, date).

### 3. Update `README.md` versioned-releases table

Add a row for the new version with scope and status.

### 4. Tag and push

```bash
git checkout main
git pull --rebase
# replace X.Y.Z accordingly
git tag -a vX.Y.Z -m "release vX.Y.Z — short description"
git push origin vX.Y.Z
```

### 5. Publish the release on GitHub

1. Go to <https://github.com/ShahnawazKakarh/speech-emotion-recognition-transfer-learning/releases/new>.
2. Select the tag you just pushed.
3. Title: `vX.Y.Z — <short description>` (e.g. `v2.0.0 — multi-seed cross-lingual paper`).
4. Description: bullet list of what's new. Paste the paper abstract for paper-bumping releases.
5. **Attach artifacts** for paper-bumping releases: the paper PDF, the aggregate
   results JSON, any trained checkpoint that's small enough to host (otherwise
   link to HuggingFace).
6. Click **Publish release**.

Zenodo will receive the webhook within ~60 seconds and create the deposit.

### 6. Wait for the DOI and update references

1. After ~1–2 minutes, refresh <https://zenodo.org/me/uploads>. The new deposit appears with status `Published`.
2. Copy the **version DOI** from the Zenodo record.
3. If this is the first auto-linked release, also copy the **Concept DOI** (shown above the version DOI under "Cite all versions").
4. Update `CITATION.cff` with the new DOIs.
5. Update `README.md`:
   - the DOI badge points at the **Concept DOI**;
   - the versioned-releases table row gets the version DOI as a hyperlink.
6. Commit and push:

   ```bash
   git commit -am "docs: update DOIs for vX.Y.Z release"
   git push origin main
   ```

7. **ORCID**: log into <https://orcid.org/my-orcid> and click **Sync from Zenodo** to add the new deposit to your Works list. If the sync is already automated, verify the new entry appears.

---

## Release checklist (copy this into each release PR description)

```
- [ ] CITATION.cff version + date-released bumped
- [ ] README versioned-releases table updated
- [ ] Tag pushed (vX.Y.Z)
- [ ] GitHub release published with artifacts attached
- [ ] Zenodo deposit confirmed (version DOI noted)
- [ ] Concept DOI noted (first auto-release only)
- [ ] README badge + CITATION.cff updated with new DOIs
- [ ] ORCID synced
- [ ] arXiv comment updated if preprint is hosted there
```
