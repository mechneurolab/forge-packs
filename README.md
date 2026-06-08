# forge-packs

The curated catalog of **forgepacks** for [FORGE Studio](https://github.com/mechneurolab/forge-studio) — installable reconstruction packs (workflows + backend modules + colormaps).

FORGE Studio's **Settings → Packs → Get forgepacks** reads `index.json` from this repo's `main` branch and installs packs from the GitHub Releases linked in it.

```
index.json                       # the catalog (this is what the app fetches)
schema/pack-registry-v1.json     # JSON Schema for index.json (CI-validated)
schema/pack-manifest-v1.json     # manifest schema (referenced by the registry schema)
packs/<id>/                      # pack sources (manifest + modules) for review
scripts/validate-index.mjs       # validates index.json + verifies each asset's sha256
```

## How a pack gets published

1. An author runs **Submit a pack** in FORGE Studio, which exports a `.forgepack`, writes a candidate `index.json` entry, and opens a pre-filled issue here (label `pack-submission`) with the entry + the file to attach.
2. A maintainer reviews the pack's code (it runs on users' machines) and commits the sources under `packs/<id>/` via a normal PR — that review is the trust gate.
3. The maintainer runs the **Publish pack** workflow (Actions → *Publish pack* → *Run workflow*, enter the pack id, optional `tags`). It builds the `.forgepack`, cuts a GitHub Release with it, computes `sha256` + `size`, adds the entry to `index.json` (`verified: true`), validates (schema + re-downloaded asset checksum), and opens a PR. To publish manually instead, see the steps in `.github/workflows/publish-pack.yml`.
4. On merge, `Validate catalog` re-verifies every entry's `sha256` against its released asset and fires the docs deploy hook — the pack appears in everyone's catalog and at [forge-mri.dev/studio/catalog](https://forge-mri.dev/studio/catalog).

> **One-time setup** for the Publish pack workflow: enable *Settings → Actions → General → Allow GitHub Actions to create and approve pull requests*, so the workflow can open the index PR with the built-in token (no PAT needed).

## Catalog entry shape

Each entry in `index.json` is a superset of the pack manifest plus distribution metadata. The authoritative schema is `schema/pack-registry-v1.json`; required fields: `id, name, version, author, modules, scripts, download_url, sha256, size, verified`. See FORGE Studio's `specs/Phase8/forge-pack-registry-spec.md` §3 for a worked example.

## Catalog website

The catalog is also browsable at **[forge-mri.dev/studio/catalog](https://forge-mri.dev/studio/catalog)** — a section of the FORGE Studio docs (VitePress) that reads this `index.json` at **build time** and renders a searchable grid plus a detail page per pack.

Because the data is baked in at build time, the website only picks up new packs when the studio docs rebuild. The `Validate catalog` workflow fires a **Cloudflare Pages deploy hook** on every push to `main` to trigger that rebuild.

**One-time setup** (maintainer): in the Cloudflare dashboard, open the `forge-studio` Pages project → **Settings → Builds & deployments → Deploy hooks**, create a hook (e.g. `forge-packs-update`) on the production branch, copy its URL, and add it here as the repo secret **`STUDIO_DOCS_DEPLOY_HOOK`** (`Settings → Secrets and variables → Actions`). Until it's set, the trigger step skips harmlessly and the catalog still refreshes on the next unrelated studio docs deploy.

## Security

- The app verifies the downloaded archive's `sha256` against `index.json` **before extraction** and only fetches from GitHub release hosts. CI re-verifies checksums on every PR.
- Curation (review of `packs/<id>/`) is the trust gate — packs execute MATLAB/Python/Julia code.
