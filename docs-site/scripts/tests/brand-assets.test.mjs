import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const read = (file) => fs.readFileSync(path.join(root, file), 'utf8');
const shape = (svg) => svg.match(/<g\b[\s\S]*<\/g>/)?.[0].replace(/\s+/g, ' ').trim();

test('GitHub and identity banners use the current site mark and palette', () => {
  const banner = read('../assets/openquant-banner.svg');
  const mark = read('src/assets/openquant-mark.svg');
  assert.ok(shape(mark), 'The site mark has a reusable group');
  assert.equal(shape(banner), shape(mark), 'Banner mark must follow the site mark');
  assert.equal(banner, read('../docs/design/identity/openquant-banner.svg'));
  assert.match(banner, /viewBox="0 0 1280 320"/);
  assert.match(banner, /aria-labelledby="banner-title banner-desc"/);
  assert.match(banner, /<title id="banner-title">OpenQuant<\/title>/);
  assert.match(banner, /<desc id="banner-desc">[^<]+<\/desc>/);
  const palette = ['#fbf9f4', '#171717', '#525252', '#cdc4b1', '#97196a', '#e44de0'];
  for (const color of banner.match(/#[\da-f]{6}/gi) || []) assert.ok(palette.includes(color), color);
  for (const color of ['#171717', '#525252', '#97196a']) assert.ok(banner.includes(color), color);
  assert.doesNotMatch(banner, /<(?:script|image|foreignObject)\b|https?:\/\/(?!www\.w3\.org)/);
});
