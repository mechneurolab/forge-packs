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
2. A maintainer reviews the pack's code (it runs on users' machines), then:
   - adds the pack sources under `packs/<id>/`,
   - cuts a **GitHub Release** with the reviewed `.forgepack` as an asset,
   - adds the entry to `index.json` with the asset's `download_url`, `sha256`, `size`, and `verified: true`,
   - opens a PR. CI (`Validate catalog`) checks `index.json` against the schema **and** re-verifies every entry's `sha256` against its released asset.
3. On merge, the pack appears in everyone's catalog on the next refresh.

## Catalog entry shape

Each entry in `index.json` is a superset of the pack manifest plus distribution metadata. The authoritative schema is `schema/pack-registry-v1.json`; required fields: `id, name, version, author, modules, scripts, download_url, sha256, size, verified`. See FORGE Studio's `specs/Phase8/forge-pack-registry-spec.md` §3 for a worked example.

## Security

- The app verifies the downloaded archive's `sha256` against `index.json` **before extraction** and only fetches from GitHub release hosts. CI re-verifies checksums on every PR.
- Curation (review of `packs/<id>/`) is the trust gate — packs execute MATLAB/Python/Julia code.
