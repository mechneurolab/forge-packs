// scripts/build-entry.mjs
// Build (or replace) a single pack's index.json entry from its committed
// sources under packs/<id>/ plus the distribution metadata of a published
// Release asset. Driven by the publish-pack GitHub workflow; runnable locally.
//
// Usage:
//   node scripts/build-entry.mjs \
//     --id <pack-id> \
//     --download-url <release asset URL> \
//     --sha256 <hex> \
//     --size <bytes> \
//     --published-at <ISO 8601> \
//     [--tags a,b,c]
//
// Reads packs/<id>/manifest.yaml, derives modules/scripts from the bundled
// files, merges the entry into index.json (replacing any existing entry with
// the same id), and writes index.json back. The result is then schema- and
// checksum-validated by `npm run validate`.

import { readFileSync, writeFileSync, existsSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';
import yaml from 'js-yaml';

function arg(name, { required = false } = {}) {
  const i = process.argv.indexOf(`--${name}`);
  if (i === -1 || i === process.argv.length - 1) {
    if (required) {
      console.error(`✗ missing required --${name}`);
      process.exit(1);
    }
    return undefined;
  }
  return process.argv[i + 1];
}

const id = arg('id', { required: true });
const downloadUrl = arg('download-url', { required: true });
const sha256 = arg('sha256', { required: true });
const size = Number(arg('size', { required: true }));
const publishedAt = arg('published-at');
const tagsArg = arg('tags');

const packDir = path.join('packs', id);
const manifestPath = path.join(packDir, 'manifest.yaml');
if (!existsSync(manifestPath)) {
  console.error(`✗ no manifest at ${manifestPath} — is the pack committed under packs/${id}/?`);
  process.exit(1);
}
if (!Number.isInteger(size) || size < 0) {
  console.error(`✗ --size must be a non-negative integer (got ${arg('size')})`);
  process.exit(1);
}

const manifest = yaml.load(readFileSync(manifestPath, 'utf8'));
if (!manifest || manifest.id !== id) {
  console.error(`✗ manifest.id (${manifest?.id}) does not match --id (${id})`);
  process.exit(1);
}

// Collect executable module files (.m/.py/.jl) under the backend dirs, as
// archive-relative paths (e.g. "matlab/prepEmpire.m"). Mirrors the app's export.
const MODULE_DIRS = ['matlab', 'python', 'julia'];
const MODULE_EXT = new Set(['.m', '.py', '.jl']);
function walk(dir, base) {
  const out = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const abs = path.join(dir, entry.name);
    const rel = path.posix.join(base, entry.name);
    if (entry.isDirectory()) out.push(...walk(abs, rel));
    else if (MODULE_EXT.has(path.extname(entry.name))) out.push(rel);
  }
  return out;
}
const scripts = [];
for (const d of MODULE_DIRS) {
  const abs = path.join(packDir, d);
  if (existsSync(abs) && statSync(abs).isDirectory()) scripts.push(...walk(abs, d));
}
scripts.sort();

const modules = {
  matlab: scripts.some((s) => s.startsWith('matlab/')),
  python: scripts.some((s) => s.startsWith('python/')),
  julia: scripts.some((s) => s.startsWith('julia/')),
};

const tags = tagsArg
  ? tagsArg.split(',').map((t) => t.trim()).filter(Boolean)
  : undefined;

// Field order mirrors the app's buildSubmissionEntry, with the maintainer-only
// distribution fields (download_url/sha256/size/verified/published_at) appended.
const entry = {
  id: manifest.id,
  name: manifest.name,
  version: manifest.version,
  author: manifest.author,
  ...(manifest.license ? { license: manifest.license } : {}),
  ...(manifest.description ? { description: manifest.description } : {}),
  ...(tags && tags.length ? { tags } : {}),
  ...(manifest.min_studio_version ? { min_studio_version: manifest.min_studio_version } : {}),
  ...(manifest.requires ? { requires: manifest.requires } : {}),
  modules,
  scripts,
  download_url: downloadUrl,
  sha256,
  size,
  verified: true,
  ...(publishedAt ? { published_at: publishedAt } : {}),
};

const indexPath = 'index.json';
const index = JSON.parse(readFileSync(indexPath, 'utf8'));
if (!Array.isArray(index.packs)) {
  console.error('✗ index.json has no packs array');
  process.exit(1);
}
const existing = index.packs.findIndex((p) => p.id === id);
if (existing !== -1) index.packs.splice(existing, 1);
index.packs.push(entry);
index.packs.sort((a, b) => a.name.localeCompare(b.name));

writeFileSync(indexPath, JSON.stringify(index, null, 2) + '\n');
console.log(
  `✓ ${existing !== -1 ? 'updated' : 'added'} ${id} v${manifest.version} ` +
    `(${scripts.length} script(s), ${size} bytes) in index.json`,
);
