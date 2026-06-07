---
name: Pack submission
about: Propose a forgepack for the catalog
title: 'Pack submission: <name> v<version>'
labels: pack-submission
---

<!--
FORGE Studio's "Submit a pack" flow pre-fills this for you (entry + checklist)
and exports the .forgepack + a <id>.entry.json next to it — attach both below.
If submitting manually, paste your candidate index.json entry in the block.
-->

## Candidate `index.json` entry

A maintainer adds `download_url` + `verified` when cutting the Release.

```json
{
}
```

## Checklist
- [ ] Attached the exported `.forgepack` (and `<id>.entry.json`).
- [ ] Pack id is unique and kebab-case.
- [ ] The pack's code has been reviewed for safety (it runs on users' machines).
- [ ] A maintainer cuts a Release from the attached pack, fills in `download_url` + `sha256`, and adds the entry to `index.json`.
