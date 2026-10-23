import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import test from 'node:test';
import { worldFromWow } from '../../src/viewer/regionData';
import { TILE_SIZE } from '../../src/formats/adt';
import { chunks } from '../../src/formats/chunks';
import type { WdtTile } from '../../src/formats/wdt';
import {
	aggroRadiusYd, canonicalJson, readVmangos, cellRowOfNorth, creatureEncounter, displayFromWorld, doodadInVmangosVmap, entryAlternatives, exportArena, factionReaction,
	localFromWorld, MAP_ORIGIN, markedCells, m2Magic, mh2oCells, northOfCellRow, placementFromWorld, planTiles, reactionsFor, serverObjectVmangosLos,
	serverObjectYaw, terrainChunkTriangles, tileBox, tilesForBox, transformPoint, TRI, validateSpec, vmapGroupSkipReason, vmapTriangleCollides,
	westOfCellCol, wmoGroupExtras, wmoLiquidTiles, worldBoxOf, worldFromDisplay, worldFromLocal, worldFromPlacement, worldFromSpawnModel, yawOfMatrix,
	type ArenaSpec, type EncounterAlternative, type Vec3,
} from '../exportArena';

const close = (a: readonly number[], b: readonly number[], tol = 1e-9) => a.every((v, i) => Math.abs(v - b[i]) <= tol);

function lcg(seed: number): () => number {
	let s = seed >>> 0;
	return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 2 ** 32);
}

test('display transform equals the Explorer worldFromWow and round-trips for any continent offset', () => {
	const rand = lcg(7);
	for (let i = 0; i < 500; i++) {
		const w: Vec3 = [-9000 + rand() * 400, -300 + rand() * 300, 50 + rand() * 80];
		const offsetX = Math.floor(rand() * 20) - 10, offsetY = Math.floor(rand() * 20) - 10;
		const explorer = worldFromWow({ name: 'x', mapId: 0, wdt: 0, offsetX, offsetY }, w[0], w[1], w[2]);
		const mine = displayFromWorld(w, offsetX, offsetY);
		assert.ok(close(mine, [explorer.x, explorer.y, explorer.z], 1e-9), `${mine} vs ${explorer.toArray()}`);
		assert.ok(close(worldFromDisplay(mine, offsetX, offsetY), w, 1e-9));
		assert.ok(close(worldFromPlacement(placementFromWorld(w)), w, 1e-9));
		const anchor: Vec3 = [-8864, -192, 80];
		assert.ok(close(worldFromLocal(localFromWorld(w, anchor), anchor), w, 1e-9));
	}
});

test('a north step lowers Ze, a west step lowers Xe and an up step raises Ye, each alone', () => {
	const base: Vec3 = [-8864, -192, 80];
	const d = (step: Vec3) => displayFromWorld([base[0] + step[0], base[1] + step[1], base[2] + step[2]], 0, 0).map((v, i) => v - displayFromWorld(base, 0, 0)[i]);
	assert.ok(close(d([1, 0, 0]), [0, 0, -1]));
	assert.ok(close(d([0, 1, 0]), [-1, 0, 0]));
	assert.ok(close(d([0, 0, 1]), [0, 1, 0]));
	assert.deepEqual(localFromWorld([base[0] + 2, base[1] - 3, base[2] + 4], base), [2, -3, 4]);
});

test('terrain cell edges follow the 1600/3/128 yard grid around the map centre', () => {
	assert.equal(northOfCellRow(4096), 0);
	assert.ok(Math.abs(northOfCellRow(0) - MAP_ORIGIN) < 1e-9);
	assert.ok(Math.abs(northOfCellRow(48 * 128) - (MAP_ORIGIN - 48 * TILE_SIZE)) < 1e-9);
	for (const g of [0, 1, 6143, 6144, 8191]) {
		assert.equal(cellRowOfNorth(northOfCellRow(g) - 1e-6), g);
		assert.equal(cellRowOfNorth(northOfCellRow(g) + 1e-6), g - 1);
	}
	assert.equal(westOfCellCol, northOfCellRow);
});

test('vmangos WMO triangle rule: collision flag, or rendered non-detail face', () => {
	assert.equal(vmapTriangleCollides(0x00), false);
	assert.equal(vmapTriangleCollides(0x08), true);
	assert.equal(vmapTriangleCollides(0x20), true);
	assert.equal(vmapTriangleCollides(0x24), false);
	assert.equal(vmapTriangleCollides(0x04), false);
	assert.equal(vmapTriangleCollides(0x2c), true);
	assert.equal(vmapTriangleCollides(0x01 | 0x02 | 0x10 | 0x40 | 0x80), false);
	assert.equal(vmapGroupSkipReason(0x80, ''), 'unreachable');
	assert.equal(vmapGroupSkipReason(0x4000000, ''), 'antiportal');
	assert.equal(vmapGroupSkipReason(0, 'antiportal'), 'antiportal_name');
	assert.equal(vmapGroupSkipReason(0x2a05, 'hall'), null);
});

test('triangle flags split vmangos WMO and M2 vmap membership; bit 32 is retired', () => {
	assert.equal(TRI.VMAP_WMO, 512);
	assert.equal(TRI.VMAP_M2, 1024);
	assert.equal(TRI.SERVER_LOS_BLOCKER, 2048);
	assert.ok(!(Object.values(TRI) as number[]).includes(32), 'v1 LOS_VMAP_STATIC must not come back');
	const bits = Object.values(TRI) as number[];
	assert.equal(new Set(bits).size, bits.length);
	for (const b of bits) assert.equal(b & (b - 1), 0, `${b} is one bit`);
});

test('server objects and vmangos unit LoS: model construction, GO state and CanAlwaysBreakLoS', () => {
	const data = (over: Record<number, number> = {}) => Array.from({ length: 24 }, (_, i) => over[i] ?? 0);
	const blocks = (type: number, state: number, d = data()) => serverObjectVmangosLos({ type, state, data: d }).blocks_unit_los;
	assert.equal(blocks(0, 1), true, 'a closed door (GO_STATE_READY) blocks');
	assert.equal(blocks(0, 0), false, 'an open door (GO_STATE_ACTIVE) has its model disabled');
	assert.equal(blocks(5, 1), true, 'generic');
	assert.equal(blocks(7, 1), false, 'chair: MOD_M2, ignored with ignoreM2Model');
	assert.equal(blocks(3, 1), false, 'chest: MOD_M2');
	const serverOnly = serverObjectVmangosLos({ type: 5, state: 1, data: data({ 2: 1 }) });
	assert.deepEqual([serverOnly.model, serverOnly.blocks_unit_los], [false, false], 'GameObjectModel::construct skips server-only generic objects');
	assert.equal(serverObjectVmangosLos({ type: 6, state: 1, data: data({ 8: 1 }) }).model, false, 'server-only trap');
	assert.equal(serverObjectVmangosLos({ type: 8, state: 1, data: data({ 3: 1 }) }).model, false, 'server-only spell focus');
	assert.equal(serverObjectVmangosLos({ type: 1, state: 1, data: data({ 8: 1 }) }).model, false, 'button with losOK');
	assert.equal(serverObjectVmangosLos({ type: 10, state: 1, data: data({ 16: 1 }) }).model, false, 'goober with losOK');
	assert.equal(serverObjectVmangosLos({ type: 10, state: 1, data: data() }).model, true);
	assert.equal(m2Magic(new TextEncoder().encode('MD21....')), 'MD21');
	assert.equal(m2Magic(new TextEncoder().encode('MD20....')), 'MD20');
	assert.equal(m2Magic(new TextEncoder().encode('MVER....')), null);
});

test('faction reaction follows FactionTemplate IsHostileTo/IsFriendlyTo against the player templates', () => {
	const human = { faction: 1, group: 3, friend: 2, enemy: 12, enemies: [0, 0, 0, 0], friends: [0, 0, 0, 0] };
	const orc = { faction: 2, group: 5, friend: 4, enemy: 10, enemies: [0, 0, 0, 0], friends: [0, 0, 0, 0] };
	const monster = { faction: 14, group: 8, friend: 0, enemy: 1, enemies: [0, 0, 0, 0], friends: [0, 0, 0, 0] };
	assert.equal(factionReaction(monster, human), 'hostile');
	assert.equal(factionReaction(monster, orc), 'hostile');
	const kobold = { faction: 25, group: 8, friend: 0, enemy: 0, enemies: [0, 0, 0, 0], friends: [25, 0, 0, 0] };
	assert.equal(factionReaction(kobold, human), 'neutral', 'Northshire kobolds: monster group, no enemy group');
	const stormwind = { faction: 72, group: 2, friend: 2, enemy: 4, enemies: [0, 0, 0, 0], friends: [72, 0, 0, 0] };
	assert.equal(factionReaction(stormwind, human), 'friendly');
	assert.equal(factionReaction(stormwind, orc), 'hostile');
	assert.equal(factionReaction({ ...kobold, enemies: [1, 0, 0, 0] }, human), 'hostile', 'an Enemies[] entry naming the player faction wins');
	assert.equal(factionReaction({ ...monster, friends: [2, 0, 0, 0] }, orc), 'friendly', 'a Friends[] entry naming the player faction wins over the enemy group');
	assert.equal(factionReaction({ ...kobold, friend: 2 }, human), 'friendly');
});

function chunk(holes = new Uint8Array(8), height = (r: number, c: number, inner: boolean) => (inner ? 10.5 : 10) + r * 0 + c * 0) {
	const heights = new Float32Array(145);
	for (let r = 0; r < 9; r++) for (let c = 0; c < 9; c++) heights[r * 17 + c] = height(r, c, false);
	for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) heights[r * 17 + 9 + c] = height(r, c, true);
	return { indexX: 3, indexY: 5, heights, holes };
}
const everything = { northMin: -1e9, northMax: 1e9, westMin: -1e9, westMax: 1e9 };

test('client terrain fan: four triangles per cell, holes removed, corners on the cell grid, normals up', () => {
	const tris = terrainChunkTriangles(32, 48, chunk(), everything);
	assert.equal(tris.length, 256);
	const holes = new Uint8Array(8);
	holes[2] = 0b1001;
	assert.equal(terrainChunkTriangles(32, 48, chunk(holes), everything).length, 256 - 8);
	const first = tris[0];
	assert.ok(close(first.a, [northOfCellRow(48 * 128 + 5 * 8), westOfCellCol(32 * 128 + 3 * 8), 10]));
	assert.ok(close(first.b, [northOfCellRow(48 * 128 + 40) - 25 / 12, westOfCellCol(32 * 128 + 24) - 25 / 12, 10.5]));
	assert.equal(first.local, 0);
	assert.equal(tris[255].local, 255);
	for (const { a, b, c } of terrainChunkTriangles(32, 48, chunk(undefined, (r, c2, inner) => r * 0.7 - c2 * 0.3 + (inner ? 0.2 : 0)), everything)) {
		const e1 = [b[0] - a[0], b[1] - a[1]], e2 = [c[0] - a[0], c[1] - a[1]];
		assert.ok(e1[0] * e2[1] - e1[1] * e2[0] > 0);
	}
	const corner = northOfCellRow(48 * 128 + 40);
	const narrow = { northMin: corner - 1, northMax: corner, westMin: -1e9, westMax: 1e9 };
	assert.equal(terrainChunkTriangles(32, 48, chunk(), narrow).length, 8 * 3);
});

function groupBytes(sub: [string, Uint8Array][]): Uint8Array {
	const header = 68;
	const body = sub.reduce((n, [, d]) => n + 8 + d.length, 0);
	const out = new Uint8Array(8 + header + body);
	const view = new DataView(out.buffer);
	out.set([...'PGOM'].map((ch) => ch.charCodeAt(0)), 0);
	view.setUint32(4, header + body, true);
	view.setUint32(8, 42, true);
	let at = 8 + header;
	for (const [id, data] of sub) {
		out.set([...id].reverse().map((ch) => ch.charCodeAt(0)), at);
		view.setUint32(at + 4, data.length, true);
		out.set(data, at + 8);
		at += 8 + data.length;
	}
	return out;
}

test('WMO polygon flags read from MOPY and from MPY2, with collision-only materials and MODR', () => {
	const a = wmoGroupExtras(groupBytes([['MOPY', new Uint8Array([0x20, 3, 0x08, 0xff, 0x24, 1])], ['MODR', new Uint8Array([5, 0, 9, 0])]]));
	assert.equal(a.polyFormat, 'MOPY');
	assert.deepEqual([...a.flags], [0x20, 0x08, 0x24]);
	assert.deepEqual([...a.invisible], [0, 1, 0]);
	assert.deepEqual(a.modr, [5, 9]);
	assert.equal(a.nameOffset, 42);
	const mpy2 = new Uint8Array(8);
	new DataView(mpy2.buffer).setUint16(0, 0x0108, true);
	new DataView(mpy2.buffer).setUint16(2, 0xffff, true);
	new DataView(mpy2.buffer).setUint16(4, 0x0020, true);
	new DataView(mpy2.buffer).setUint16(6, 7, true);
	const b = wmoGroupExtras(groupBytes([['MPY2', mpy2]]));
	assert.equal(b.polyFormat, 'MPY2');
	assert.deepEqual([...b.flags], [0x0108, 0x0020]);
	assert.deepEqual([...b.invisible], [1, 0]);
	assert.throws(() => wmoGroupExtras(groupBytes([['MOPY', new Uint8Array(2)], ['MPY2', new Uint8Array(4)]])), /more than one/);
	assert.throws(() => wmoGroupExtras(groupBytes([['MPY2', new Uint8Array(6)]])), /multiple of 4/);
	assert.equal(wmoGroupExtras(groupBytes([])).polyFormat, null);
});

test('model yaw and server spawn orientation use 0 north, pi/2 west', () => {
	const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
	assert.ok(Math.abs(yawOfMatrix(identity) - 1.5 * Math.PI) < 1e-12, 'placement +X is east');
	const towardSouth = [0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1];
	assert.ok(Math.abs(yawOfMatrix(towardSouth) - Math.PI) < 1e-12);
	assert.ok(close(transformPoint(identity, 1, 2, 3), [1, 2, 3]));
	assert.ok(close(worldFromSpawnModel(1, 0, 0, 100, 200, 5, 0, 1), [101, 200, 5]));
	assert.ok(close(worldFromSpawnModel(1, 0, 0, 100, 200, 5, Math.PI / 2, 2), [100, 202, 5]));
	assert.ok(close(worldFromSpawnModel(0, 1, 0.5, 100, 200, 5, 0, 1), [100, 201, 5.5]), 'model +Y (left) is west when facing north');
});

test('canonical JSON sorts keys, normalises -0 and rejects non-finite numbers', () => {
	assert.equal(canonicalJson({ b: 1, a: [true, null, 'x', -0], c: { z: 1.5, y: undefined } }), '{"a":[true,null,"x",0],"b":1,"c":{"z":1.5}}');
	assert.throws(() => canonicalJson({ a: Number.NaN }), /Non-finite/);
	assert.throws(() => canonicalJson([Infinity]), /Non-finite/);
});

const rules = { threat_radius_yd: 50, leash_extension_timer_s: 12, runner_speed_yd_s: 7, player_level_min: 5 };
const spec: ArenaSpec = {
	spec_version: 'wow-arena-spec-v3', arena_id: 'test-arena', description: '', product: 'wow_classic_beta', map_id: 0, map_directory: 'Azeroth', wdt_fdid: 775971,
	anchor_world: { north: -8864, west: -192, up: 80 }, bounds_local: { x_min: -80, x_max: 80, y_min: -64, y_max: 64 }, margin_yd: 4, tile_search_margin_yd: 64,
	spawn_margin_yd: 64, encounter_rules: rules,
};

test('arena specs are validated and map to the expected world box and tile', () => {
	validateSpec(spec);
	assert.deepEqual(worldBoxOf(spec), { northMin: -8944, northMax: -8784, westMin: -256, westMax: -128 });
	assert.deepEqual(tilesForBox(worldBoxOf(spec)), [[32, 48]]);
	assert.deepEqual(tilesForBox({ northMin: -8540, northMax: -8530, westMin: -5, westMax: 5 }), [[31, 47], [32, 47], [31, 48], [32, 48]]);
	assert.throws(() => validateSpec({ ...spec, spec_version: 'wow-arena-spec-v2' }), /Unsupported/);
	assert.throws(() => validateSpec({ ...spec, bounds_local: { ...spec.bounds_local, x_min: 90 } }), /reversed/);
	assert.throws(() => validateSpec({ ...spec, anchor_world: { ...spec.anchor_world, up: Number.NaN } }), /finite/);
	assert.throws(() => validateSpec({ ...spec, arena_id: 'Bad Id' }), /arena_id/);
	assert.throws(() => validateSpec({ ...spec, tile_search_margin_yd: 1 }), /Margins/);
	assert.throws(() => validateSpec({ ...spec, spawn_margin_yd: -1 }), /spawn_margin_yd/);
	assert.throws(() => validateSpec({ ...spec, encounter_rules: { ...rules, threat_radius_yd: 0 } }), /encounter_rules/);
	assert.throws(() => validateSpec({ ...spec, encounter_rules: { ...rules, player_level_min: 0 } }), /encounter_rules/);
});

const wdtTile = (x: number, y: number, obj0 = 1000 + x, root = 2000 + x): WdtTile => ({ x, y, flags: 0, flowMap: 0, files: { root, obj0, obj1: 0, tex0: 0, lod: 0, mapTexture: 0, mapTextureN: 0, minimap: 0 } });

test('tile planning fails closed: a missing tile or obj0 inside the geometry box is an error, outside it is declared', () => {
	const geometry = { northMin: -9955, northMax: -9627, westMin: -14, westMax: 314 };
	const search = { northMin: -10015, northMax: -9567, westMin: -74, westMax: 374 };
	const all = new Array<WdtTile | null>(64 * 64).fill(null);
	for (const [x, y] of [[31, 49], [32, 49], [31, 50], [32, 50]]) all[y * 64 + x] = wdtTile(x, y);
	assert.deepEqual(planTiles(all, search, geometry), { present: [[31, 49], [32, 49], [31, 50], [32, 50]], absentOutsideGeometry: [] });
	const marginGone = all.slice();
	marginGone[49 * 64 + 31] = null;
	assert.deepEqual(planTiles(marginGone, search, geometry), { present: [[32, 49], [31, 50], [32, 50]], absentOutsideGeometry: [[31, 49]] });
	const insideGone = all.slice();
	insideGone[50 * 64 + 31] = null;
	assert.throws(() => planTiles(insideGone, search, geometry), /Missing ADT tile 31_50 overlaps the arena geometry box/);
	const noObj0 = all.slice();
	noObj0[49 * 64 + 32] = wdtTile(32, 49, 0);
	assert.throws(() => planTiles(noObj0, search, geometry), /ADT tile 32_49 has no obj0/);
	const noRoot = all.slice();
	noRoot[50 * 64 + 32] = wdtTile(32, 50, 5, 0);
	assert.throws(() => planTiles(noRoot, search, geometry), /ADT tile 32_50 has no root/);
	const tb = tileBox(32, 48);
	assert.ok(close([tb.northMax, tb.westMax], [MAP_ORIGIN - 48 * TILE_SIZE, MAP_ORIGIN - 32 * TILE_SIZE]));
});

test('WMO liquid tiles: only tiles whose low nibble is not 0xF, transformed by the placement matrix', () => {
	const cell = TILE_SIZE / 128;
	const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
	const liquid = { xVerts: 3, yVerts: 2, xTiles: 2, yTiles: 1, position: [100, 200, 5] as [number, number, number], heights: new Float32Array([5, 5, 5, 6, 6, 6]), tiles: new Uint8Array([0x0f, 0x40]) };
	const tiles = wmoLiquidTiles(liquid, identity);
	assert.equal(tiles.length, 1);
	assert.deepEqual(tiles[0].tile, [1, 0]);
	// WMO space is z-up: tile (1, 0) spans x in [100 + cell, 100 + 2 cell], y in [200, 200 + cell]; heights 5 and 6.
	const corners = [[100 + cell, 200, 5], [100 + 2 * cell, 200, 5], [100 + cell, 200 + cell, 6], [100 + 2 * cell, 200 + cell, 6]].map((p) => worldFromPlacement(transformPoint(identity, p[0], p[1], p[2])));
	assert.ok(close([tiles[0].box.northMin, tiles[0].box.northMax], [Math.min(...corners.map((c) => c[0])), Math.max(...corners.map((c) => c[0]))]));
	assert.ok(close([tiles[0].box.westMin, tiles[0].box.westMax], [Math.min(...corners.map((c) => c[1])), Math.max(...corners.map((c) => c[1]))]));
	assert.ok(close([tiles[0].upMin, tiles[0].upMax], [Math.min(...corners.map((c) => c[2])), Math.max(...corners.map((c) => c[2]))]));
	assert.equal(tiles[0].liquidNibble, 0);
	assert.throws(() => wmoLiquidTiles({ ...liquid, tiles: new Uint8Array([0]) }, identity), /MLIQ/);
});

test('cells overlapped by a world box, clipped to the coverage grid', () => {
	const layout = { row0: 6000, col0: 4000, rows: 4, cols: 3 };
	const inside = { northMin: northOfCellRow(6002) + 0.1, northMax: northOfCellRow(6001) - 0.1, westMin: westOfCellCol(4001) + 0.1, westMax: westOfCellCol(4000) - 0.1 };
	assert.deepEqual(markedCells(inside, layout), [[1, 0]]);
	const twoCols = { ...inside, westMin: westOfCellCol(4002) + 0.1 };
	assert.deepEqual(markedCells(twoCols, layout), [[1, 0], [1, 1]]);
	assert.equal(markedCells({ northMin: -1e9, northMax: 1e9, westMin: -1e9, westMax: 1e9 }, layout).length, 12);
	assert.deepEqual(markedCells({ northMin: 1e8, northMax: 1e9, westMin: 0, westMax: 1 }, layout), []);
});

// ---------------------------------------------------------------------------------------------
// Exporter error paths against the real local install, with injected faults (test-only hooks).

const EXPLORER = resolve(import.meta.dirname, '..', '..');
const SPECS = resolve(EXPLORER, '..', 'combat_lab_gpu', 'arenas', 'specs');
const SAFE_LOCS = resolve(EXPLORER, '..', '.scratch', 'sandbox-program', 'T1', 'sources', 'WorldSafeLocs-1.13.2.31446.csv');
const VMANGOS_DB = resolve(EXPLORER, '..', '.scratch', 'sandbox-program', 'T1', 'sources', 'vmangos-db', 'mangos.sqlite');
/** The private local exports the golden tests compare a fresh export against (git-ignored, never committed). */
const ARENA_DATA = resolve(EXPLORER, '..', 'combat_lab_gpu', 'arenas', 'data');
const WOW_DIR = 'C:/Program Files (x86)/World of Warcraft';

function scratchArgs(specName: string, spawns = 'public/spawns/map0.json') {
	const out = mkdtempSync(join(tmpdir(), 'arena-test-'));
	return { args: { spec: join(SPECS, specName), out, wowDir: WOW_DIR, spawns: join(EXPLORER, spawns), safeLocs: SAFE_LOCS, vmangosDb: VMANGOS_DB, report: null }, out };
}

test('real install: the specs and the install this suite needs are present', () => {
	assert.ok(existsSync(join(WOW_DIR, '.build.info')), 'the local WoW install is missing');
	assert.ok(existsSync(join(SPECS, 'northshire-abbey-north-v3.json')) && existsSync(join(SPECS, 'elwynn-fargodeep-v2.json')));
	assert.ok(existsSync(VMANGOS_DB), 'the vmangos world DB snapshot is missing');
});

test('real install: a missing ADT tile inside the arena is a hard export error', async () => {
	const { args, out } = scratchArgs('northshire-abbey-north-v3.json');
	try {
		await assert.rejects(exportArena(args, { wdt: (w) => ({ ...w, tiles: w.tiles.map((t) => (t && t.x === 32 && t.y === 48 ? null : t)) }) }), /Missing ADT tile 32_48/);
		assert.ok(!existsSync(join(out, 'northshire-abbey-north-v3')));
	} finally { rmSync(out, { recursive: true, force: true }); }
});

test('real install: a tile without obj0 is a hard export error', async () => {
	const { args, out } = scratchArgs('northshire-abbey-north-v3.json');
	try {
		await assert.rejects(exportArena(args, { wdt: (w) => ({ ...w, tiles: w.tiles.map((t) => (t && t.x === 32 && t.y === 48 ? { ...t, files: { ...t.files, obj0: 0 } } : t)) }) }), /has no obj0/);
	} finally { rmSync(out, { recursive: true, force: true }); }
});

test('real install: terrain chunks missing inside the bounds fail the export (no unknown coverage)', async () => {
	const { args, out } = scratchArgs('northshire-abbey-north-v3.json');
	try {
		await assert.rejects(exportArena(args, { adt: (_x, _y, adt) => ({ ...adt, chunks: adt.chunks.filter((c) => !(c.indexX === 5 && c.indexY === 9)) }) }), /coverage cells inside the bounds have no terrain data/);
	} finally { rmSync(out, { recursive: true, force: true }); }
});

function setGroupFlag(bytes: Uint8Array, bit: number): Uint8Array {
	const copy = bytes.slice();
	const view = new DataView(copy.buffer);
	for (const c of chunks(copy)) if (c.id === 'MOGP') view.setUint32(c.offset + 8, view.getUint32(c.offset + 8, true) | bit, true);
	return copy;
}

/** Appends an MLIQ chunk (2x2 vertices, one water tile near the group's bounding-box minimum) inside MOGP. */
function withLiquid(bytes: Uint8Array): Uint8Array {
	const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	let mogp = { offset: 0, size: 0 };
	for (const c of chunks(bytes)) if (c.id === 'MOGP') mogp = c;
	const min = [view.getFloat32(mogp.offset + 0x0c, true), view.getFloat32(mogp.offset + 0x10, true), view.getFloat32(mogp.offset + 0x14, true)];
	const body = new Uint8Array(30 + 4 * 8 + 1);
	const b = new DataView(body.buffer);
	b.setUint32(0, 2, true); b.setUint32(4, 2, true); b.setUint32(8, 1, true); b.setUint32(12, 1, true);
	b.setFloat32(16, min[0] + 1, true); b.setFloat32(20, min[1] + 1, true); b.setFloat32(24, min[2] + 0.5, true);
	for (let i = 0; i < 4; i++) b.setFloat32(30 + i * 8 + 4, min[2] + 0.5, true);
	body[30 + 32] = 0x00;
	const end = mogp.offset + mogp.size;
	const out = new Uint8Array(bytes.length + 8 + body.length);
	out.set(bytes.subarray(0, end), 0);
	out.set([...'QILM'].map((ch) => ch.charCodeAt(0)), end);
	new DataView(out.buffer).setUint32(end + 4, body.length, true);
	out.set(body, end + 8);
	out.set(bytes.subarray(end), end + 8 + body.length);
	const ov = new DataView(out.buffer);
	ov.setUint32(mogp.offset - 4, mogp.size + 8 + body.length, true);
	ov.setUint32(mogp.offset + 8, ov.getUint32(mogp.offset + 8, true) | 0x1000, true);
	return out;
}

test('real install: a WMO group flagged as liquid without MLIQ fails the export', async () => {
	const { args, out } = scratchArgs('northshire-abbey-north-v3.json');
	let patched = 0;
	try {
		await assert.rejects(exportArena(args, { file: (_f, role, bytes) => (role === 'wmo_group' && patched++ === 0 ? setGroupFlag(bytes, 0x1000) : bytes) }), /liquid flag 0x1000 but no MLIQ/);
	} finally { rmSync(out, { recursive: true, force: true }); }
});

test('real install: WMO liquid (MLIQ) is labelled in coverage and listed, never exported as dry floor', async () => {
	const { args, out } = scratchArgs('northshire-abbey-north-v3.json');
	let patched = 0;
	try {
		const result = await exportArena(args, { file: (_f, role, bytes) => (role === 'wmo_group' && patched++ === 0 ? withLiquid(bytes) : bytes) });
		const manifest = JSON.parse(readFileSync(join(result.dir, 'manifest.json'), 'utf8'));
		assert.deepEqual(manifest.test_fault_injection, ['file']);
		assert.equal(manifest.wmo_liquids.length, 1);
		assert.ok(manifest.wmo_liquids[0].cells_marked > 0);
		const coverage = readFileSync(join(result.dir, 'coverage.bin'));
		assert.equal(manifest.counts.coverage_wmo_liquid_cells, [...coverage].filter((v) => v & 8).length);
		assert.ok(manifest.counts.coverage_wmo_liquid_cells >= manifest.wmo_liquids[0].cells_marked);
	} finally { rmSync(out, { recursive: true, force: true }); }
});

test('real install: ADT MH2O liquid cells are labelled in coverage, never dry floor', async () => {
	const { args, out } = scratchArgs('northshire-abbey-north-v3.json');
	try {
		// Chunk (5, 9) of tile 32_48 lies inside the bounds; one 8x8 liquid instance covers it.
		const liquid = { chunk: 9 * 16 + 5, type: 1, minHeight: 60, maxHeight: 60, x: 0, y: 0, width: 8, height: 8, exists: null, heights: null };
		const result = await exportArena(args, { adt: (x, y, adt) => (x === 32 && y === 48 ? { ...adt, liquids: [...adt.liquids, liquid] } : adt) });
		const manifest = JSON.parse(readFileSync(join(result.dir, 'manifest.json'), 'utf8'));
		assert.equal(manifest.liquids.length, 1);
		assert.equal(manifest.liquids[0].cells_in_arena, 64);
		const coverage = readFileSync(join(result.dir, 'coverage.bin'));
		assert.equal([...coverage].filter((v) => v & 4).length, 64);
		assert.equal(manifest.counts.coverage_liquid_cells, 64);
		// The cell containing the anchor (local 0, 0) is one of them.
		const { row0_global: row0, col0_global: col0, cols } = manifest.coverage;
		const r = cellRowOfNorth(-8864) - row0, c = cellRowOfNorth(-192) - col0;
		assert.ok(coverage[r * cols + c] & 4);
	} finally { rmSync(out, { recursive: true, force: true }); }
});

test('real install: legacy MCLQ liquid flags without MH2O fail the export', async () => {
	const { args, out } = scratchArgs('northshire-abbey-north-v3.json');
	try {
		await assert.rejects(exportArena(args, { adt: (x, y, adt) => (x === 32 && y === 48 ? { ...adt, chunkIds: adt.chunkIds.filter((id) => id !== 'MH2O'),
			chunks: adt.chunks.map((c) => (c.indexX === 5 && c.indexY === 9 ? { ...c, flags: c.flags | 0x04 } : c)) } : adt) }), /legacy MCLQ liquid/);
	} finally { rmSync(out, { recursive: true, force: true }); }
});

/** Clears the file-ID flag of the first MDDF row, so the placement names its model by path. */
function firstMddfByName(obj0: Uint8Array): Uint8Array {
	const copy = obj0.slice();
	const view = new DataView(copy.buffer);
	for (const c of chunks(copy)) {
		if (c.id === 'MDDF' && c.size >= 36) {
			view.setUint16(c.offset + 34, view.getUint16(c.offset + 34, true) & ~0x40, true);
			return copy;
		}
	}
	throw new Error('obj0 has no MDDF row');
}

test('real install: a placement that names its model by path fails the export', async () => {
	const { args, out } = scratchArgs('northshire-abbey-north-v3.json');
	try {
		await assert.rejects(exportArena(args, { file: (_f, role, bytes) => (role === 'adt_obj0_32_48' ? firstMddfByName(bytes) : bytes) }), /1 placements reference models by name/);
	} finally { rmSync(out, { recursive: true, force: true }); }
});

test('vmangos vmap keeps a WMO doodad only from the placement set and only when an extracted group references it', () => {
	assert.equal(doodadInVmangosVmap(1, 1, true), true);
	assert.equal(doodadInVmangosVmap(0, 1, true), false, 'set 0 is drawn by the client but not extracted when the placement uses another set');
	assert.equal(doodadInVmangosVmap(1, 1, false), false, 'unreferenced doodads are not extracted');
	assert.equal(doodadInVmangosVmap(0, 0, true), true);
});

test('real install (golden): a fresh export of the pinned Fargodeep arena reproduces its pinned geometry, spawns and evidence', async () => {
	const { args, out } = scratchArgs('elwynn-fargodeep-v2.json');
	try {
		const result = await exportArena(args);
		const pinnedDir = join(ARENA_DATA, 'elwynn-fargodeep-v2');
		const sha = (dir: string, name: string) => createHash('sha256').update(readFileSync(join(dir, name))).digest('hex');
		const spawns = JSON.parse(readFileSync(join(result.dir, 'spawns.json'), 'utf8'));
		const byGuid = new Map(spawns.creatures.map((c: { guid: number }) => [c.guid, c]));
		const spider = byGuid.get(80653) as { entry_choice: { alternatives: { entry: number; probability: number; reaction: { alliance: string } }[] } };
		assert.deepEqual(spider.entry_choice.alternatives.map((a) => [a.entry, a.probability, a.reaction.alliance]), [[30, 0.5, 'hostile'], [113, 0.5, 'neutral']],
			'guid 80653 is Forest Spider or Stonetusk Boar (creature.id2)');
		const miner = byGuid.get(80628) as { entry_choice: { alternatives: { entry: number; detection_range_yd: number }[] } };
		assert.deepEqual(miner.entry_choice.alternatives.map((a) => [a.entry, a.detection_range_yd]), [[40, 18], [475, 18]]);
		const geometry = JSON.parse(readFileSync(join(result.dir, 'geometry.json'), 'utf8'));
		assert.equal(geometry.server_object_pools['8002'].max_limit, 3);
		assert.equal(geometry.server_object_pools['8002'].objects.length, 20);
		assert.equal(geometry.server_object_pools['3393'].mother_pool, 3424);
		const flags = new Uint16Array(readFileSync(join(result.dir, 'tri_flags.bin')).buffer.slice(0));
		const source = new Uint32Array(readFileSync(join(result.dir, 'tri_source.bin')).buffer.slice(0));
		geometry.sources.forEach((s: { kind: string; collision?: string }, i: number) => {
			if (s.kind !== 'server_gameobject') return;
			const conditional = s.collision === 'conditional-pool';
			for (let k = 0; k < flags.length; k++) if (source[k] === i) assert.equal(Boolean(flags[k] & TRI.CONDITIONAL_POOL), conditional);
		});
		assert.ok(geometry.sources.some((s: { collision?: string }) => s.collision === 'conditional-pool'));
		for (const f of ['geometry.json', 'spawns.json', 'evidence.json']) {
			assert.equal(sha(result.dir, f), sha(pinnedDir, f), `${f} drifted from the pinned export`);
		}
		const manifest = JSON.parse(readFileSync(join(result.dir, 'manifest.json'), 'utf8'));
		// The whole manifest must equal the pinned one except exporter.tool_sha256, the one allow-listed field: the pinned export
		// was written by an exporter revision that was edited afterwards (round-3 finding 3). Both values are reported.
		const pinnedManifest = JSON.parse(readFileSync(join(pinnedDir, 'manifest.json'), 'utf8'));
		const freshTool = manifest.exporter.tool_sha256, pinnedTool = pinnedManifest.exporter.tool_sha256;
		console.log(`golden manifest: exporter.tool_sha256 pinned ${pinnedTool}, fresh ${freshTool}`);
		assert.equal(freshTool, createHash('sha256').update(readFileSync(resolve(EXPLORER, 'tools', 'exportArena.ts'))).digest('hex'));
		assert.deepEqual({ ...manifest, exporter: { ...manifest.exporter, tool_sha256: null } }, { ...pinnedManifest, exporter: { ...pinnedManifest.exporter, tool_sha256: null } },
			'the fresh manifest differs from the pinned one in more than exporter.tool_sha256');
		assert.equal(Object.keys(manifest.exporter).sort().join(','), 'explorer_head,explorer_status_porcelain,imported_source_sha256,node,tool,tool_sha256');
		assert.ok(manifest.counts.not_in_vmangos_vmap_triangles > 0);
		const outside = manifest.sources.filter((s: { kind: string; vmangos_vmap?: boolean }) => s.kind === 'm2_wmo_doodad' && s.vmangos_vmap === false);
		assert.ok(outside.length > 0);
		const inside = manifest.sources.filter((s: { kind: string; vmangos_vmap?: boolean }) => s.kind === 'm2_wmo_doodad' && s.vmangos_vmap === true);
		assert.ok(inside.length > 0);
	} finally { rmSync(out, { recursive: true, force: true }); }
});

test('real install (golden): elwynn-east-v1 (v7 evidence region) re-exports byte for byte, manifest included', async () => {
	const { args, out } = scratchArgs('elwynn-east-v1.json');
	try {
		const result = await exportArena(args);
		const pinnedDir = join(ARENA_DATA, 'elwynn-east-v1');
		const sha = (dir: string, name: string) => createHash('sha256').update(readFileSync(join(dir, name))).digest('hex');
		const toolSha = createHash('sha256').update(readFileSync(resolve(EXPLORER, 'tools', 'exportArena.ts'))).digest('hex');
		const pinnedManifest = JSON.parse(readFileSync(join(pinnedDir, 'manifest.json'), 'utf8'));
		assert.equal(pinnedManifest.exporter.tool_sha256, toolSha, 'elwynn-east-v1 was exported by this exporter revision; a changed exporter needs a new arena id');
		const lock = JSON.parse(readFileSync(join(pinnedDir, 'arena.lock.json'), 'utf8'));
		assert.equal(sha(result.dir, 'manifest.json'), lock.manifest_sha256, 'the whole manifest (exporter sha included) reproduces the pinned arena digest');
		for (const f of ['manifest.json', 'geometry.json', 'spawns.json', 'evidence.json', 'triangles.bin', 'tri_flags.bin', 'tri_source.bin', 'tri_local.bin', 'coverage.bin']) {
			assert.equal(sha(result.dir, f), sha(pinnedDir, f), `${f} drifted from the pinned export`);
		}
	} finally { rmSync(out, { recursive: true, force: true }); }
});

test('real install: a missing tile only in the search margin is declared, not silently skipped', async () => {
	const { args, out } = scratchArgs('elwynn-fargodeep-v2.json');
	try {
		// The v1 box (+-160 yd around the same anchor): tile 31_49 overlaps only its search margin.
		const base = JSON.parse(readFileSync(args.spec, 'utf8'));
		const specPath = join(out, 'margin-spec.json');
		writeFileSync(specPath, JSON.stringify({ ...base, arena_id: 'margin-test', bounds_local: { x_min: -160, x_max: 160, y_min: -160, y_max: 160 } }));
		const result = await exportArena({ ...args, spec: specPath }, { wdt: (w) => ({ ...w, tiles: w.tiles.map((t) => (t && t.x === 31 && t.y === 49 ? null : t)) }) });
		const manifest = JSON.parse(readFileSync(join(result.dir, 'manifest.json'), 'utf8'));
		assert.deepEqual(manifest.map.tiles, { present: [[32, 49], [31, 50], [32, 50]], absent_outside_geometry: [[31, 49]] });
		assert.deepEqual(manifest.test_fault_injection, ['wdt']);
		assert.equal(manifest.counts.coverage_unknown_cells, 0);
	} finally { rmSync(out, { recursive: true, force: true }); }
});

// ---------------------------------------------------------------------------------------------
// Encounter data (spawns-v2) and server-object placement: pure functions.

test('aggro radius follows vmangos Creature::GetAttackDistance with creature_template.detection_range', () => {
	assert.equal(aggroRadiusYd(18, 6, 6), 18, 'equal level');
	assert.equal(aggroRadiusYd(18, 9, 5), 22, '1 yd per level the player is below');
	assert.equal(aggroRadiusYd(18, 5, 15), 8);
	assert.equal(aggroRadiusYd(18, 5, 40), 5, 'floor min(detection, 5)');
	assert.equal(aggroRadiusYd(3, 5, 40), 3);
	assert.equal(aggroRadiusYd(18, 60, 1), 43, 'GetAttackDistance itself: difference capped at 25 levels below (the 40 yd candidate search cap is separate)');
	assert.equal(aggroRadiusYd(0.5, 10, 1), 0, 'detection under 1 yd: no proximity aggro');
});

test('entry alternatives follow CreatureData::ChooseCreatureId: uniform over the non-zero id slots', () => {
	assert.deepEqual(entryAlternatives(1, [40, 475, 0, 0, 0]), [{ entry: 40, probability: 0.5 }, { entry: 475, probability: 0.5 }]);
	assert.deepEqual(entryAlternatives(1, [30, 0, 0, 0, 0]), [{ entry: 30, probability: 1 }]);
	assert.deepEqual(entryAlternatives(1, [30, 113, 30, 0, 0]), [{ entry: 30, probability: 2 / 3 }, { entry: 113, probability: 1 / 3 }]);
	assert.deepEqual(entryAlternatives(1, [30, 0, 113, 0, 0]), [{ entry: 30, probability: 1 }], 'vmangos stops counting at the first zero slot');
	assert.throws(() => entryAlternatives(7, [0, 0, 0, 0, 0]), /creature 7/);
});

test('reactions are computed against the Alliance template for alliance and the Horde template for horde', () => {
	const human = { faction: 1, group: 3, friend: 2, enemy: 12, enemies: [0, 0, 0, 0], friends: [0, 0, 0, 0] };
	const orc = { faction: 2, group: 5, friend: 4, enemy: 10, enemies: [0, 0, 0, 0], friends: [0, 0, 0, 0] };
	const hordeGuard = { faction: 85, group: 4, friend: 4, enemy: 2, enemies: [0, 0, 0, 0], friends: [0, 0, 0, 0] };
	assert.deepEqual(reactionsFor(hordeGuard, human, orc), { alliance: 'hostile', horde: 'friendly' });
	assert.deepEqual(reactionsFor(null, human, orc), { alliance: 'unknown', horde: 'unknown' });
});

const alt = (over: Partial<EncounterAlternative> = {}): EncounterAlternative => ({
	entry: 40, probability: 1, name: 'Kobold Miner', subname: '', level_min: 6, level_max: 7, rank: 0, creature_type: 7, npc_flags: 0, faction_template: 26,
	reaction: { alliance: 'hostile', horde: 'hostile' }, detection_range_yd: 18, call_for_help_range_yd: 5, leash_range_yd: 0, speed_walk: 1, speed_run: 0.857143, ...over,
});

test('encounter fields: farthest-waypoint roam, max aggro over alternatives, threat area, leash envelope and bounds', () => {
	const box = { anchor_world: { north: 0, west: 0, up: 0 }, bounds_local: { x_min: -200, x_max: 200, y_min: -200, y_max: 200 }, encounter_rules: rules };
	// Waypoints 30 yd and then 10 yd from the spawn: roam is the farthest (30), not the last.
	const path = [30, 0, 0, 0, 10, 0, 0, 0];
	const e = creatureEncounter({ guid: 9, world: [0, 0, 0], orientation: 0, movement: -1, path, alternatives: [alt({ probability: 0.5 }), alt({ entry: 475, level_max: 9, probability: 0.5, reaction: { alliance: 'neutral', horde: 'hostile' } })], respawn: [270, 270], group: null }, box);
	const m = e.movement as { roam_radius_yd: number; kind: string; path_source: string };
	assert.equal(m.roam_radius_yd, 30);
	assert.deepEqual([m.kind, m.path_source], ['waypoints', 'guid']);
	const aggro = e.aggro as { aggro_radius_max_yd: number; threat_area_radius_yd: number };
	assert.equal(aggro.aggro_radius_max_yd, 22, 'level 9 alternative at player level 5: 18 + 4');
	assert.equal(aggro.threat_area_radius_yd, 50, 'max(1.5 * 22, ThreatRadius 50)');
	assert.deepEqual(e.hostile_probability, { alliance: 0.5, horde: 1 });
	assert.equal(e.engagement_radius_yd, 30 + 22 + 50);
	assert.equal(e.leash_envelope_yd, 30 + 22 + 50 + 84, 'runner 7 yd/s for the 12 s leash timer');
	assert.equal(e.edge_distance_yd, 200);
	assert.deepEqual([e.encounter_admissible, e.leash_contained], [true, true]);
	const at = (x: number) => creatureEncounter({ guid: 9, world: [x, 0, 0], orientation: 0, movement: 4, path: null, alternatives: [alt()], respawn: [270, 270], group: null }, box);
	// roam 4, aggro 20 (level 7 vs 5), threat area 50: engagement 74; leash envelope 158.
	assert.equal(at(200 - 74).encounter_admissible, true, 'edge exactly the engagement radius');
	assert.equal(at(200 - 73.9).encounter_admissible, false);
	assert.equal(at(200 - 158).leash_contained, true);
	assert.equal(at(200 - 157.9).leash_contained, false);
	// A wide detection range makes 1.5 x aggro exceed ThreatRadius: level 9 vs player 5 with 40 yd gives 44, threat area 66.
	const wide = creatureEncounter({ guid: 9, world: [0, 0, 0], orientation: 0, movement: 0, path: null, alternatives: [alt({ detection_range_yd: 40, level_max: 9 })], respawn: [270, 270], group: null }, box);
	assert.deepEqual([(wide.aggro as { aggro_radius_max_yd: number }).aggro_radius_max_yd, (wide.aggro as { threat_area_radius_yd: number }).threat_area_radius_yd, wide.engagement_radius_yd], [44, 66, 110]);
	const outside = at(205);
	assert.deepEqual([outside.in_bounds, outside.encounter_admissible], [false, false], 'a spawn in the spawn margin is outside the bounds');
});

test('server object rotation must be the yaw of its orientation (yaw-only quaternions)', () => {
	const o = 1.2;
	const near = (a: number, b: number) => Math.abs(a - b) < 1e-12;
	assert.ok(near(serverObjectYaw(5, [0, 0, Math.sin(o / 2), Math.cos(o / 2)], o), o));
	assert.ok(near(serverObjectYaw(5, [0, 0, 0, 0], o), o), 'an all-zero rotation defers to the orientation');
	assert.ok(Math.abs(serverObjectYaw(5, [0, 0, Math.sin(-o / 2), Math.cos(-o / 2)], -o) - (2 * Math.PI - o)) < 1e-12, 'normalised to [0, 2 pi)');
	assert.throws(() => serverObjectYaw(5, [0, 0, Math.sin(0.4), Math.cos(0.4)], o), /Game object 5/);
	assert.throws(() => serverObjectYaw(5, [0.1, 0, Math.sin(o / 2), Math.cos(o / 2)], o), /Game object 5/);
});

test('MH2O instances mark only the cells their exists bitmap sets', () => {
	const layout = { row0: 48 * 128, col0: 32 * 128, rows: 16, cols: 16 };
	const exists = new Uint8Array(4);
	exists[0] = 1;
	exists[3] = 1;
	const base = { chunk: 0, x: 2, y: 1, width: 2, height: 2, exists: null as Uint8Array | null };
	assert.deepEqual(mh2oCells(base, 32, 48, layout), [1 * 16 + 2, 1 * 16 + 3, 2 * 16 + 2, 2 * 16 + 3]);
	assert.deepEqual(mh2oCells({ ...base, exists }, 32, 48, layout), [1 * 16 + 2, 2 * 16 + 3]);
	assert.deepEqual(mh2oCells({ ...base, chunk: 17 }, 32, 48, layout), [9 * 16 + 10, 9 * 16 + 11, 10 * 16 + 10, 10 * 16 + 11], 'chunk (1, 1) starts 8 cells in');
	assert.deepEqual(mh2oCells({ ...base, chunk: 255 }, 32, 48, layout), [], 'cells outside the coverage grid are clipped');
});

// ---------------------------------------------------------------------------------------------
// vmangos DB reader against a small sqlite fixture (no install needed).

function vmangosFixture(dir: string, edit: (db: DatabaseSync) => void = () => {}): string {
	const path = join(dir, `vm-${Math.random().toString(36).slice(2)}.sqlite`);
	const db = new DatabaseSync(path);
	const tpl = Array.from({ length: 24 }, (_, i) => `data${i} INTEGER DEFAULT 0`).join(', ');
	db.exec(`
		CREATE TABLE creature (guid INTEGER, id INTEGER, id2 INTEGER DEFAULT 0, id3 INTEGER DEFAULT 0, id4 INTEGER DEFAULT 0, id5 INTEGER DEFAULT 0, map INTEGER,
			position_x REAL, position_y REAL, position_z REAL, spawntimesecsmin INTEGER DEFAULT 270, spawntimesecsmax INTEGER DEFAULT 270, spawn_flags INTEGER DEFAULT 0,
			patch_min INTEGER DEFAULT 0, patch_max INTEGER DEFAULT 10);
		CREATE TABLE gameobject (guid INTEGER, id INTEGER, map INTEGER, position_x REAL, position_y REAL, position_z REAL, state INTEGER DEFAULT 1, spawn_flags INTEGER DEFAULT 0,
			patch_min INTEGER DEFAULT 0, patch_max INTEGER DEFAULT 10);
		CREATE TABLE game_event_creature (guid INTEGER, event INTEGER);
		CREATE TABLE game_event_gameobject (guid INTEGER, event INTEGER);
		CREATE TABLE creature_template (entry INTEGER, patch INTEGER DEFAULT 0, name TEXT, level_min INTEGER, level_max INTEGER, detection_range REAL DEFAULT 18);
		CREATE TABLE gameobject_template (entry INTEGER, patch INTEGER DEFAULT 0, type INTEGER, ${tpl});
		CREATE TABLE pool_creature (guid INTEGER, pool_entry INTEGER, chance REAL, patch_min INTEGER DEFAULT 0, patch_max INTEGER DEFAULT 10);
		CREATE TABLE pool_creature_template (id INTEGER, pool_entry INTEGER, chance REAL, patch_min INTEGER DEFAULT 0, patch_max INTEGER DEFAULT 10);
		CREATE TABLE creature_linking (guid INTEGER, master_guid INTEGER);
		CREATE TABLE creature_groups (leader_guid INTEGER, member_guid INTEGER, flags INTEGER);
		CREATE TABLE pool_gameobject (guid INTEGER, pool_entry INTEGER, chance REAL, flags INTEGER DEFAULT 0, patch_min INTEGER DEFAULT 0, patch_max INTEGER DEFAULT 10);
		CREATE TABLE pool_gameobject_template (id INTEGER, pool_entry INTEGER, chance REAL, flags INTEGER DEFAULT 0, patch_min INTEGER DEFAULT 0, patch_max INTEGER DEFAULT 10);
		CREATE TABLE pool_template (entry INTEGER, max_limit INTEGER, description TEXT, flags INTEGER DEFAULT 0, patch_min INTEGER DEFAULT 0, patch_max INTEGER DEFAULT 10);
		CREATE TABLE pool_pool (pool_id INTEGER, mother_pool INTEGER, chance REAL, flags INTEGER DEFAULT 0);
		INSERT INTO creature (guid, id, id2, map, position_x, position_y, position_z) VALUES (1, 40, 475, 0, 10, 20, 5);
		INSERT INTO creature_template (entry, name, level_min, level_max) VALUES (40, 'Kobold Miner', 6, 7), (475, 'Kobold Tunneler', 5, 6);
		INSERT INTO gameobject (guid, id, map, position_x, position_y, position_z) VALUES (7, 1731, 0, 12, 22, 5), (8, 1731, 0, 900, 900, 5), (9, 3000, 0, 14, 24, 5);
		INSERT INTO gameobject_template (entry, type) VALUES (1731, 3), (3000, 3);
		INSERT INTO pool_gameobject_template (id, pool_entry, chance) VALUES (1731, 50, 0);
		INSERT INTO pool_gameobject (guid, pool_entry, chance) VALUES (9, 60, 0);
		INSERT INTO pool_template (entry, max_limit, description) VALUES (50, 1, 'veins by template'), (60, 1, 'one object'), (70, 2, 'mother');
		INSERT INTO pool_pool (pool_id, mother_pool, chance) VALUES (60, 70, 0);
	`);
	edit(db);
	db.close();
	return path;
}

const fixtureBox = { northMin: 0, northMax: 50, westMin: 0, westMax: 50 };
const creature1 = [{ guid: 1, entry: 40, north: 10, west: 20, up: 5 }];
const objects79 = [{ guid: 7, entry: 1731, north: 12, west: 22, up: 5 }, { guid: 9, entry: 3000, north: 14, west: 24, up: 5 }];

test('vmangos reader: alternative ids, template-based and guid pools with their mother, all members also outside the arena', () => {
	const dir = mkdtempSync(join(tmpdir(), 'vmfix-'));
	try {
		const vm = readVmangos(vmangosFixture(dir), 0, creature1, fixtureBox, objects79, fixtureBox);
		assert.deepEqual(vm.creatures.get(1)!.ids, [40, 475, 0, 0, 0]);
		assert.deepEqual(vm.objects.get(7)!.pool, { entry: 50, chance: 0 }, 'pool_gameobject_template membership by entry');
		assert.deepEqual(vm.pools['50'].objects.map((o) => o.guid), [7, 8], 'members by template include the spawn outside the arena');
		assert.deepEqual([vm.pools['60'].mother_pool, vm.pools['70'].child_pools], [70, [{ pool: 60, chance: 0 }]]);
		assert.deepEqual(Object.keys(vm.pools), ['50', '60', '70']);
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});

test('vmangos reader: refuses a map0.json spawn the DB disagrees with, pooled or linked creatures and unknown spawns', () => {
	const dir = mkdtempSync(join(tmpdir(), 'vmfix-'));
	try {
		const read = (edit?: (db: DatabaseSync) => void, creatures = creature1, objects = objects79) => () => readVmangos(vmangosFixture(dir, edit), 0, creatures, fixtureBox, objects, fixtureBox);
		assert.throws(read(undefined, [{ ...creature1[0], north: 10.01 }]), /creature 1: map0.json spawn/);
		assert.throws(read(undefined, [{ ...creature1[0], entry: 475 }]), /creature 1: map0.json spawn/);
		assert.throws(read(undefined, creature1, [{ ...objects79[0], up: 5.02 }, objects79[1]]), /game object 7: map0.json spawn/);
		assert.throws(read((db) => db.exec('INSERT INTO pool_creature (guid, pool_entry, chance) VALUES (1, 5, 0)')), /pooled/);
		assert.throws(read((db) => db.exec('INSERT INTO creature_linking (guid, master_guid) VALUES (1, 2)')), /linked/);
		assert.throws(read((db) => db.exec('INSERT INTO creature (guid, id, map, position_x, position_y, position_z) VALUES (2, 40, 0, 30, 30, 5)')), /only in DB 2/);
		assert.throws(read((db) => db.exec('INSERT INTO pool_gameobject (guid, pool_entry, chance) VALUES (7, 60, 0)')), /more than one pool/);
		assert.doesNotThrow(read((db) => db.exec('INSERT INTO creature (guid, id, map, position_x, position_y, position_z, patch_min) VALUES (2, 40, 0, 30, 30, 5, 11)')), 'a later-patch spawn is not in patch 10');
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});

test('vmangos reader: pools fail closed (CheckPoolAndChance, CheckPool, member, pool_pool and template flags, spawn_flags); chances are not coerced', () => {
	const dir = mkdtempSync(join(tmpdir(), 'vmfix-'));
	try {
		const read = (edit?: (db: DatabaseSync) => void) => () => readVmangos(vmangosFixture(dir, edit), 0, creature1, fixtureBox, objects79, fixtureBox);
		// vmangos CheckPoolAndChance (PoolManager.cpp:653-667) skips rows with chance < 0 or > 100; the exporter refuses them.
		assert.throws(read((db) => db.exec('UPDATE pool_gameobject SET chance = -5 WHERE guid = 9')), /pool 60: member 9 has chance -5 outside \[0, 100\]/);
		assert.throws(read((db) => db.exec('UPDATE pool_gameobject_template SET chance = 150')), /chance 150 outside \[0, 100\]/);
		assert.throws(read((db) => db.exec('UPDATE pool_pool SET chance = -1')), /pool 70: child pool 60 has chance -1 outside \[0, 100\]/);
		// CheckPool (PoolManager.cpp:171-182, 1022-1029): explicit chances summing to neither 0 nor 100 without an equal-chance member
		// drop POOL_FLAG_AUTO_SPAWN, so the pool never spawns; the exporter refuses it.
		assert.throws(read((db) => db.exec('UPDATE pool_gameobject SET chance = 40 WHERE guid = 9')), /pool 60 fails vmangos CheckPool/);
		assert.throws(read((db) => db.exec('UPDATE pool_pool SET chance = 30; UPDATE pool_template SET max_limit = 1 WHERE entry = 70')), /pool 70 fails vmangos CheckPool/);
		// PoolGroup::AddEntry: a non-zero chance is explicit only in a max_limit 1 pool; with max_limit 2 the member is equal-chanced.
		assert.doesNotThrow(read((db) => db.exec('UPDATE pool_pool SET chance = 30')), 'mother pool 70 has max_limit 2: its chanced child is equal-chanced');
		assert.doesNotThrow(read((db) => db.exec('UPDATE pool_gameobject SET chance = 40 WHERE guid = 9; UPDATE pool_template SET max_limit = 2 WHERE entry = 60')));
		const mixed = readVmangos(vmangosFixture(dir, (db) => db.exec(`UPDATE pool_gameobject SET chance = 40.5 WHERE guid = 9;
			INSERT INTO gameobject (guid, id, map, position_x, position_y, position_z) VALUES (10, 3000, 0, 700, 700, 5);
			INSERT INTO pool_gameobject (guid, pool_entry, chance) VALUES (10, 60, 0)`)), 0, creature1, fixtureBox, objects79, fixtureBox);
		assert.deepEqual(mixed.pools['60'].objects, [{ guid: 9, chance: 40.5 }, { guid: 10, chance: 0 }], 'explicit plus equal members pass CheckPool; chances are kept as stored');
		assert.deepEqual(mixed.objects.get(9)!.pool, { entry: 60, chance: 40.5 });
		const full = readVmangos(vmangosFixture(dir, (db) => db.exec(`UPDATE pool_gameobject SET chance = 100 WHERE guid = 9`)), 0, creature1, fixtureBox, objects79, fixtureBox);
		assert.equal(full.pools['60'].objects[0].chance, 100, 'explicit chances summing to 100 pass');
		// Flags the sampler does not model: member spawn flags (FLAG_SPAWN_ENABLE_IF_WORLD_POP_OVER_BLIZZLIKE, PoolManager.cpp:36-43), pool_pool
		// flags, pool_template flags (POOL_FLAG_MAXLIMIT_SCALING_LINEAR), gameobject and creature spawn_flags.
		assert.throws(read((db) => db.exec('UPDATE pool_gameobject SET flags = 1')), /pool 60: member 9 has flags 1/);
		assert.throws(read((db) => db.exec('UPDATE pool_gameobject_template SET flags = 1')), /pool 50: member 7 has flags 1/);
		assert.throws(read((db) => db.exec('UPDATE pool_pool SET flags = 1')), /pool 70: child pool 60 has flags 1/);
		assert.throws(read((db) => db.exec('UPDATE pool_template SET flags = 2 WHERE entry = 50')), /pool_template 50 has flags 2/);
		assert.throws(read((db) => db.exec('UPDATE gameobject SET spawn_flags = 8 WHERE guid = 7')), /game object 7 has spawn_flags 8/);
		assert.throws(read((db) => db.exec('UPDATE creature SET spawn_flags = 1 WHERE guid = 1')), /creature 1 has spawn_flags 1/);
	} finally {
		rmSync(dir, { recursive: true, force: true });
	}
});
