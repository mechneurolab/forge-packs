// scripts/validate-index.mjs
// Validates index.json against the registry schema, then re-verifies every
// entry's sha256 + size against its released download_url asset. Used by CI.

import Ajv from 'ajv';
import addFormats from 'ajv-formats';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

const ajv = new Ajv({ allErrors: true, strict: false });
addFormats(ajv);
// The registry schema $refs the manifest schema's `requires`; load both.
ajv.addSchema(JSON.parse(readFileSync('schema/pack-manifest-v1.json', 'utf8')));
const validate = ajv.compile(JSON.parse(readFileSync('schema/pack-registry-v1.json', 'utf8')));

const index = JSON.parse(readFileSync('index.json', 'utf8'));

if (!validate(index)) {
  console.error('✗ index.json failed schema validation:');
  for (const e of validate.errors ?? []) console.error(`  ${e.instancePath || '/'}: ${e.message}`);
  process.exit(1);
}
console.log(`✓ index.json is schema-valid — ${index.packs.length} pack(s).`);

// Guard against duplicate ids.
const ids = index.packs.map((p) => p.id);
const dupes = ids.filter((id, i) => ids.indexOf(id) !== i);
if (dupes.length) {
  console.error(`✗ duplicate pack ids: ${[...new Set(dupes)].join(', ')}`);
  process.exit(1);
}

// Verify each released asset matches its declared sha256 + size.
let failed = false;
for (const entry of index.packs) {
  try {
    const res = await fetch(entry.download_url, { redirect: 'follow' });
    if (!res.ok) {
      console.error(`✗ ${entry.id}: download_url returned HTTP ${res.status}`);
      failed = true;
      continue;
    }
    const buf = Buffer.from(await res.arrayBuffer());
    const sha = createHash('sha256').update(buf).digest('hex');
    if (sha !== entry.sha256) {
      console.error(`✗ ${entry.id}: sha256 mismatch (declared ${entry.sha256}, asset ${sha})`);
      failed = true;
    } else if (buf.length !== entry.size) {
      console.error(`✗ ${entry.id}: size mismatch (declared ${entry.size}, asset ${buf.length})`);
      failed = true;
    } else {
      console.log(`✓ ${entry.id}: asset verified (${sha.slice(0, 12)}…, ${buf.length} bytes)`);
    }
  } catch (e) {
    console.error(`✗ ${entry.id}: ${e instanceof Error ? e.message : String(e)}`);
    failed = true;
  }
}

process.exit(failed ? 1 : 0);
