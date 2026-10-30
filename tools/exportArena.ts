// Exports one bounded arena (terrain, WMO and M2 collision, coverage, liquids, binding evidence, spawns)
// from a local World of Warcraft install into a versioned, digest-pinned directory for the combat sandbox.
//
// Usage (from the repository root):
//   npx tsx tools/exportArena.ts --spec <arena-spec.json> --out <directory> --vmangos-db <mangos.sqlite>
//     [--wow-dir "C:/Program Files (x86)/World of Warcraft"] [--spawns public/spawns/map0.json]
//     [--safe-locs <WorldSafeLocs.csv>] [--report <export-report.json>]
//
// --vmangos-db is the VMaNGOS world database (sqlite) that public/spawns/map0.json was built from. The
// exporter reads from it what map0.json lacks: alternative creature entries (creature.id2..id5), the
// creature_template fields of every alternative (detection range, levels, faction), object pools
// (pool_gameobject, pool_gameobject_template, pool_template, pool_pool), creature groups and game
// object state and template data. Every spawn taken from map0.json is checked against the database.
//
// Scope: private, local extraction from the operator's own install. The output contains game-derived
// geometry and must stay in an ignored folder; it is not for redistribution. Reads files only; it has
// no connection to a running game and sends no input.
//
// Fail-closed rules: a WDT tile missing inside the geometry box, a tile without root or obj0, terrain
// chunks missing inside the bounds, a WMO group flagged as liquid without MLIQ, legacy MCLQ liquid,
// name-referenced placements, a server object with a non-M2 model or a non-yaw rotation, a map0.json
// spawn the database disagrees with, pooled or linked creatures, and a creature group whose members have
// alternative entries all stop the export. Liquids are labelled in coverage (ADT MH2O and WMO MLIQ), never
// exported as dry floor. Pooled server objects are exported as conditional collision (CONDITIONAL_POOL).
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { CascStorage } from '../src/casc/storage';
import { chunks } from '../src/formats/chunks';
import { CHUNK_SIZE, parseAdtRoot, TILE_SIZE, type AdtChunk, type AdtRoot } from '../src/formats/adt';
import { parseWdt, type Wdt, type WdtTile } from '../src/formats/wdt';
import { parseWmoGroup, parseWmoRoot, WMO_LIQUID_CELL, type WmoLiquid } from '../src/formats/wmo';
import { parseM2 } from '../src/formats/m2';
import { Db2 } from '../src/formats/db2';
import { parsePlacements, type Placement } from '../src/explorer/objects';
import { compose, fromQuaternion, multiply, scaling, translation, type Mat4 } from '../src/explorer/mat4';
import { NodeSource } from './nodeSource';

export const ARENA_FORMAT = 'wow-arena-v3';
export const LOCK_FORMAT = 'wow-arena-lock-v1';
export const SPEC_FORMAT = 'wow-arena-spec-v3';
export const GEOMETRY_FORMAT = 'wow-arena-geometry-v2';
export const EVIDENCE_FORMAT = 'wow-arena-evidence-v1';
export const SPAWNS_FORMAT = 'wow-arena-spawns-v2';
/** VMaNGOS content patch the spawn data is read for (1.12; buildSpawns.ts uses the same). */
export const VMANGOS_PATCH = 10;
/** vmangos MAX_LEVEL_DIFF_FOR_AGGRO_RANGE (CreatureDefines.h:655). */
export const MAX_LEVEL_DIFF_FOR_AGGRO_RANGE = 25;
export const FRAME_ID = 'north-west-up-local-v1';
export const MAP_ORIGIN = 32 * TILE_SIZE;
/** Terrain cell edge: a tile has 16 chunks of 8 cells a side. */
export const CELL_SIZE = TILE_SIZE / 128;
/** Cells across a map: 64 tiles of 128 cells. */
const MAP_CELLS = 64 * 128;
/** Client tables read by file ID. */
const GAMEOBJECT_DISPLAY_INFO_FDID = 1266277;
const FACTION_TEMPLATE_FDID = 1361579;
/** MCNK flags of legacy MCLQ liquids (river, ocean, magma, slime); this exporter reads MH2O only. */
const MCNK_LEGACY_LIQUID = 0x04 | 0x08 | 0x10 | 0x20;

/**
 * Per-triangle flags (uint16). Bit 32 was wow-arena-v1's LOS_VMAP_STATIC, which wrongly put M2 collision
 * into the vmangos LoS set; it is retired and never written.
 */
export const TRI = {
	TERRAIN: 1,
	WMO: 2,
	M2_ADT: 4,
	M2_WMO_DOODAD: 8,
	MOVEMENT_SOLID: 16,
	WMO_INVISIBLE: 64,
	NOT_IN_VMANGOS_VMAP: 128,
	SERVER_GAMEOBJECT: 256,
	VMAP_WMO: 512,
	VMAP_M2: 1024,
	SERVER_LOS_BLOCKER: 2048,
	/** A pooled server object: its triangles exist only while vmangos has the object spawned. */
	CONDITIONAL_POOL: 4096,
} as const;

/** Coverage cell bits (uint8). Zero means no terrain data: unknown, never free ground. */
export const COVER = { TERRAIN: 1, HOLE: 2, LIQUID: 4, WMO_LIQUID: 8 } as const;

/** MOPY triangle flags, as named by vmangos contrib/vmap_extractor/vmapextract/wmo.h. */
export const MOPY = { DETAIL: 0x04, COLLISION: 0x08, RENDER: 0x20 } as const;
/** MOGP group flags: the two the vmangos extractor skips (WMOGroup::ShouldSkip) and the liquid flag. */
export const MOGP = { UNREACHABLE: 0x80, ANTIPORTAL: 0x4000000, LIQUID: 0x1000 } as const;

export type Vec3 = [number, number, number];

/** WoW world (x north, y west, z up) from ADT placement space (x east, y up, z south). */
export function worldFromPlacement(p: readonly number[]): Vec3 {
	return [MAP_ORIGIN - p[2], MAP_ORIGIN - p[0], p[1]];
}

export function placementFromWorld(w: readonly number[]): Vec3 {
	return [MAP_ORIGIN - w[1], w[2], MAP_ORIGIN - w[0]];
}

/** Explorer display space: placement space shifted by the viewer's continent layout offset (tiles). */
export function displayFromWorld(w: readonly number[], offsetX: number, offsetY: number): Vec3 {
	const p = placementFromWorld(w);
	return [p[0] + offsetX * TILE_SIZE, p[1], p[2] + offsetY * TILE_SIZE];
}

export function worldFromDisplay(d: readonly number[], offsetX: number, offsetY: number): Vec3 {
	return worldFromPlacement([d[0] - offsetX * TILE_SIZE, d[1], d[2] - offsetY * TILE_SIZE]);
}

export function localFromWorld(w: readonly number[], anchor: readonly number[]): Vec3 {
	return [w[0] - anchor[0], w[1] - anchor[1], w[2] - anchor[2]];
}

export function worldFromLocal(l: readonly number[], anchor: readonly number[]): Vec3 {
	return [l[0] + anchor[0], l[1] + anchor[1], l[2] + anchor[2]];
}

/** North edge of global terrain cell row g (rows run south) and west edge of column g (columns run east). */
export function northOfCellRow(g: number): number {
	return ((MAP_CELLS / 2 - g) * 25) / 6;
}
export const westOfCellCol = northOfCellRow;

export function cellRowOfNorth(north: number): number {
	return Math.floor((MAP_ORIGIN - north) / CELL_SIZE);
}
export const cellColOfWest = cellRowOfNorth;

/** vmangos WMOGroup::ConvertToVMAPGroupWmo: collision flag, or a rendered face that is not detail. */
export function vmapTriangleCollides(flags: number): boolean {
	const render = (flags & MOPY.RENDER) !== 0 && (flags & MOPY.DETAIL) === 0;
	return (flags & MOPY.COLLISION) !== 0 || render;
}

/** vmangos WMOGroup::ShouldSkip; null when the group is extracted. */
export function vmapGroupSkipReason(flags: number, name: string): string | null {
	if (flags & MOGP.UNREACHABLE) return 'unreachable';
	if (flags & MOGP.ANTIPORTAL) return 'antiportal';
	if (name === 'antiportal') return 'antiportal_name';
	return null;
}

/** 'MD21' (chunked) or 'MD20' (legacy) for an M2 file, null for anything else (such as a WMO). */
export function m2Magic(bytes: Uint8Array): 'MD21' | 'MD20' | null {
	const id = new TextDecoder().decode(bytes.subarray(0, 4));
	return id === 'MD21' || id === 'MD20' ? id : null;
}

/** GameObject types (vmangos SharedDefines.h) the LoS rule distinguishes. */
const GO_TYPE = { DOOR: 0, BUTTON: 1, CHEST: 3, GENERIC: 5, TRAP: 6, SPELL_FOCUS: 8, GOOBER: 10 } as const;

/**
 * vmangos dynamic LoS for one server object with an .m2 display model (vmangos 53d24d5c):
 * - GameObjectModel::construct (GameObjectModel.cpp:130-150) builds no model for a button or goober with
 *   losOK (data8 / data16) or a server-only generic, trap or spell focus (data2 / data8 / data3,
 *   GameObjectInfo::IsServerOnly);
 * - GameObject::UpdateCollisionState (GameObject.cpp:2350-2357) enables the model only in GO_STATE_READY
 *   (spawn state 1), or for a chest while its loot is ready (true when it spawns);
 * - .m2 models get MOD_M2 unless GameObjectInfo::CanAlwaysBreakLoS (door, generic), and unit LoS ignores
 *   MOD_M2 models (Map::isInLineOfSight ignoreM2Model = true).
 * The exporter refuses non-M2 display models, so a WMO-model object never reaches this rule.
 */
export function serverObjectVmangosLos(go: { type: number; state: number; data: readonly number[] }): { model: boolean; enabled: boolean; blocks_unit_los: boolean; reason: string } {
	const d = (i: number) => go.data[i] ?? 0;
	let noModel: string | null = null;
	if (go.type === GO_TYPE.BUTTON && d(8)) noModel = 'button with losOK';
	else if (go.type === GO_TYPE.GOOBER && d(16)) noModel = 'goober with losOK';
	else if ((go.type === GO_TYPE.GENERIC && d(2)) || (go.type === GO_TYPE.TRAP && d(8)) || (go.type === GO_TYPE.SPELL_FOCUS && d(3))) noModel = 'server-only object';
	if (noModel) return { model: false, enabled: false, blocks_unit_los: false, reason: `no dynamic model: ${noModel}` };
	const enabled = go.type === GO_TYPE.CHEST ? true : go.state === 1;
	const breaks = go.type === GO_TYPE.DOOR || go.type === GO_TYPE.GENERIC;
	const reason = !enabled ? 'model disabled: spawn state is not GO_STATE_READY' : breaks ? 'door or generic: no MOD_M2, blocks unit LoS' : '.m2 model with MOD_M2: ignored by unit LoS';
	return { model: true, enabled, blocks_unit_los: enabled && breaks, reason };
}

export interface WorldBox { northMin: number; northMax: number; westMin: number; westMax: number }

/**
 * vmangos Doodad::ExtractSet (vmapexport.cpp ExtractSingleWmo): only the placement's doodad set is extracted,
 * and only doodads that an extracted group references through MODR.
 */
export function doodadInVmangosVmap(setIndex: number, placementSet: number, referencedByExtractedGroup: boolean): boolean {
	return setIndex === placementSet && referencedByExtractedGroup;
}

/** One FactionTemplate row: field 0 Faction, 2 FactionGroup, 3 FriendGroup, 4 EnemyGroup, 5 Enemies[], 6 Friends[]. */
export interface FactionTemplateRow { faction: number; group: number; friend: number; enemy: number; enemies: number[]; friends: number[] }

/**
 * Reaction of a creature's faction template to a player template, as FactionTemplateEntry::IsHostileTo and
 * IsFriendlyTo (TrinityCore/vmangos DBCStructure.h): an Enemies[]/Friends[] entry naming the player's faction
 * decides first, then the group masks. Reputation-based reactions (factions with a reputation index) are not modelled.
 */
export function factionReaction(creature: FactionTemplateRow, player: FactionTemplateRow): 'hostile' | 'friendly' | 'neutral' {
	if (player.faction) {
		if (creature.enemies.includes(player.faction)) return 'hostile';
		if (creature.friends.includes(player.faction)) return 'friendly';
	}
	if (creature.enemy & player.group) return 'hostile';
	if ((creature.friend & player.group) || (creature.group & player.friend)) return 'friendly';
	return 'neutral';
}

export function expandBox(box: WorldBox, by: number): WorldBox {
	return { northMin: box.northMin - by, northMax: box.northMax + by, westMin: box.westMin - by, westMax: box.westMax + by };
}

export function reactionsFor(creature: FactionTemplateRow | null, alliance: FactionTemplateRow, horde: FactionTemplateRow): { alliance: string; horde: string } {
	if (!creature) return { alliance: 'unknown', horde: 'unknown' };
	return { alliance: factionReaction(creature, alliance), horde: factionReaction(creature, horde) };
}

/**
 * vmangos Creature::GetAttackDistance (Creature.cpp:2193-2238) for a player target without detect-range
 * auras: detection range (creature_template.detection_range) minus (player level - creature level), the
 * difference capped at 25 levels below, floored at min(detection, 5); no proximity aggro under 1 yd;
 * times Rate.Creature.Aggro.
 */
export function aggroRadiusYd(detectionRange: number, creatureLevel: number, playerLevel: number, rate = 1): number {
	if (rate === 0) return 0;
	let leveldif = playerLevel - creatureLevel;
	if (leveldif < -MAX_LEVEL_DIFF_FOR_AGGRO_RANGE) leveldif = -MAX_LEVEL_DIFF_FOR_AGGRO_RANGE;
	if (detectionRange < 1) return 0;
	return Math.max(detectionRange - leveldif, Math.min(detectionRange, 5)) * rate;
}

/**
 * vmangos CreatureData::ChooseCreatureId (Creature.cpp:63-76): the id slots are counted up to the first
 * zero and one slot is drawn uniformly, so an entry listed twice is twice as likely. vmangos falls back
 * to entry 1 when every slot is zero; the exporter refuses that instead.
 */
export function entryAlternatives(guid: number, ids: readonly number[]): { entry: number; probability: number }[] {
	const slots: number[] = [];
	for (const id of ids) {
		if (!id) break;
		slots.push(id);
	}
	if (!slots.length) throw new Error(`creature ${guid}: no creature id in id..id5`);
	const out: { entry: number; probability: number }[] = [];
	for (const id of slots) {
		const seen = out.find((a) => a.entry === id);
		if (seen) seen.probability += 1 / slots.length;
		else out.push({ entry: id, probability: 1 / slots.length });
	}
	return out;
}

export interface EncounterAlternative {
	entry: number; probability: number; name: string; subname: string; level_min: number; level_max: number; rank: number; creature_type: number;
	npc_flags: number; faction_template: number; reaction: { alliance: string; horde: string }; detection_range_yd: number;
	call_for_help_range_yd: number; leash_range_yd: number; speed_walk: number; speed_run: number;
}

export interface EncounterInput {
	guid: number; world: Vec3; orientation: number; movement: number; path: readonly number[] | null;
	alternatives: EncounterAlternative[]; respawn: [number, number]; group: Record<string, unknown> | null;
}

/**
 * Encounter record of one creature spawn for spawns-v2. movement > 0 is a random wander radius, -1 its
 * own waypoint path, -2 its template path (map0.json convention). Roam is the wander radius or the
 * farthest waypoint from the spawn. The aggro radius is the largest over the alternatives at the spec's
 * player_level_min; the threat area is max(1.5 * aggro, ThreatRadius) (Creature::IsOutOfThreatArea).
 * engagement = roam + aggro + threat area (a superset of where creature and victim can be before a leash);
 * the leash envelope adds runner speed x the 12 s leash timer, a lower bound because being attacked or
 * crowd-controlled re-arms the timer.
 */
export function creatureEncounter(input: EncounterInput, spec: Pick<ArenaSpec, 'anchor_world' | 'bounds_local' | 'encounter_rules'>): Record<string, unknown> {
	const { guid, world, orientation, movement, path, alternatives } = input;
	const anchor: Vec3 = [spec.anchor_world.north, spec.anchor_world.west, spec.anchor_world.up];
	const rules = spec.encounter_rules;
	const b = spec.bounds_local;
	const route: Record<string, unknown>[] = [];
	let roam = movement > 0 ? movement : 0;
	if (path) {
		for (let i = 0; i + 3 < path.length; i += 4) {
			const l = localFromWorld([path[i], path[i + 1], path[i + 2]], anchor);
			route.push({ point: i / 4, world: { north: path[i], west: path[i + 1], up: path[i + 2] }, local: { x: l[0], y: l[1], z: l[2] }, wait_s: path[i + 3],
				in_bounds: l[0] >= b.x_min && l[0] <= b.x_max && l[1] >= b.y_min && l[1] <= b.y_max });
			roam = Math.max(roam, Math.hypot(path[i] - world[0], path[i + 1] - world[1]));
		}
	}
	const l = localFromWorld(world, anchor);
	const inBounds = l[0] >= b.x_min && l[0] <= b.x_max && l[1] >= b.y_min && l[1] <= b.y_max;
	const edge = Math.min(l[0] - b.x_min, b.x_max - l[0], l[1] - b.y_min, b.y_max - l[1]);
	const aggro = Math.max(...alternatives.map((a) => aggroRadiusYd(a.detection_range_yd, a.level_max, rules.player_level_min)));
	const threat = Math.max(1.5 * aggro, rules.threat_radius_yd);
	const engage = roam + aggro + threat;
	const leashRun = rules.runner_speed_yd_s * rules.leash_extension_timer_s;
	const hostile = (side: 'alliance' | 'horde') => alternatives.filter((a) => a.reaction[side] === 'hostile').reduce((s, a) => s + a.probability, 0);
	return {
		guid, world: { north: world[0], west: world[1], up: world[2] }, local: { x: l[0], y: l[1], z: l[2] }, orientation,
		entry_choice: { rule: 'vmangos CreatureData::ChooseCreatureId: one of the non-zero creature.id..id5 slots, uniformly, at each (re)spawn', alternatives },
		hostile_probability: { alliance: hostile('alliance'), horde: hostile('horde') },
		movement: { kind: movement > 0 ? 'random' : path ? 'waypoints' : 'idle', wander_radius_yd: Math.max(movement, 0), path_source: movement === -1 ? 'guid' : movement === -2 ? 'template' : null, roam_radius_yd: roam },
		waypoints: route, respawn_s: input.respawn, group: input.group, in_bounds: inBounds, edge_distance_yd: edge,
		aggro: { player_level_min: rules.player_level_min, aggro_radius_max_yd: aggro, threat_area_radius_yd: threat },
		engagement_radius_yd: engage, leash_envelope_yd: engage + leashRun,
		encounter_admissible: inBounds && edge >= engage,
		leash_contained: inBounds && edge >= engage + leashRun,
	};
}

/**
 * Yaw of a server-spawned object. Only yaw rotations are supported: the quaternion's x and y must vanish
 * and, unless the quaternion is all zero, its yaw must match the spawn orientation within 0.01 rad.
 * Returns the orientation normalised to [0, 2 pi).
 *
 * With maxTiltRad > 0 (opt-in, spec server_object_max_tilt_rad) a rotation that tilts the object's up axis by at
 * most that angle is accepted and placed with its yaw only: the tilt is dropped, and the yaw is the quaternion's
 * heading (atan2(2(wz + xy), 1 - 2(y^2 + z^2))), which must still match the spawn orientation within 0.01 rad.
 */
export function serverObjectTilt(rotation: readonly number[]): number {
	const [r0, r1, r2, r3] = rotation;
	const n2 = r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3;
	if (n2 === 0) return 0;
	return Math.acos(Math.max(-1, Math.min(1, 1 - (2 * (r0 * r0 + r1 * r1)) / n2)));
}

export function serverObjectYaw(guid: number, rotation: readonly number[], orientation: number, maxTiltRad = 0): number {
	const [r0, r1, r2, r3] = rotation;
	if (maxTiltRad > 0 && (Math.abs(r0) >= 1e-3 || Math.abs(r1) >= 1e-3)) {
		const tilt = serverObjectTilt(rotation);
		const n2 = r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3;
		const heading = Math.atan2(2 * (r3 * r2 + r0 * r1), n2 - 2 * (r1 * r1 + r2 * r2));
		const headingDelta = Math.abs(Math.atan2(Math.sin(heading - orientation), Math.cos(heading - orientation)));
		if (tilt > maxTiltRad || headingDelta > 0.01) throw new Error(`Game object ${guid}: rotation ${[r0, r1, r2, r3]} tilts ${tilt.toFixed(3)} rad (allowed ${maxTiltRad}) or is not the yaw ${orientation}; unsupported`);
		return ((orientation % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
	}
	const yawOnly = Math.abs(r0) < 1e-3 && Math.abs(r1) < 1e-3;
	const quaternionYaw = 2 * Math.atan2(r2, r3);
	const yawDelta = Math.abs(Math.atan2(Math.sin(quaternionYaw - orientation), Math.cos(quaternionYaw - orientation)));
	if (!yawOnly || ((r2 !== 0 || r3 !== 0) && yawDelta > 0.01)) throw new Error(`Game object ${guid}: rotation ${[r0, r1, r2, r3]} is not the yaw ${orientation}; unsupported`);
	return ((orientation % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
}

/** Coverage indices (row * cols + col) of the cells one MH2O instance covers, honouring its exists bitmap. */
export function mh2oCells(liquid: { chunk: number; x: number; y: number; width: number; height: number; exists: Uint8Array | null },
	tileX: number, tileY: number, layout: { row0: number; col0: number; rows: number; cols: number }): number[] {
	const ix = liquid.chunk % 16, iy = Math.floor(liquid.chunk / 16);
	const out: number[] = [];
	for (let y = 0; y < liquid.height; y++) {
		for (let x = 0; x < liquid.width; x++) {
			if (liquid.exists && !(liquid.exists[y * liquid.width + x])) continue;
			const gr = tileY * 128 + iy * 8 + liquid.y + y - layout.row0;
			const gc = tileX * 128 + ix * 8 + liquid.x + x - layout.col0;
			if (gr < 0 || gr >= layout.rows || gc < 0 || gc >= layout.cols) continue;
			out.push(gr * layout.cols + gc);
		}
	}
	return out;
}

/** Conservative test: the triangle's horizontal bounding box overlaps the box (edges inclusive). */
export function triangleTouchesBox(a: Vec3, b: Vec3, c: Vec3, box: WorldBox): boolean {
	return Math.max(a[0], b[0], c[0]) >= box.northMin && Math.min(a[0], b[0], c[0]) <= box.northMax &&
		Math.max(a[1], b[1], c[1]) >= box.westMin && Math.min(a[1], b[1], c[1]) <= box.westMax;
}

export function pointInBox(w: readonly number[], box: WorldBox): boolean {
	return w[0] >= box.northMin && w[0] <= box.northMax && w[1] >= box.westMin && w[1] <= box.westMax;
}

function boxesOverlap(a: WorldBox, b: WorldBox): boolean {
	return a.northMax >= b.northMin && a.northMin <= b.northMax && a.westMax >= b.westMin && a.westMin <= b.westMax;
}

/** Applies a column-major 4x4 matrix (Explorer mat4 layout) in float64. */
export function transformPoint(m: ArrayLike<number>, x: number, y: number, z: number): Vec3 {
	return [
		m[0] * x + m[4] * y + m[8] * z + m[12],
		m[1] * x + m[5] * y + m[9] * z + m[13],
		m[2] * x + m[6] * y + m[10] * z + m[14],
	];
}

/**
 * World position of a model-space point (x forward, y left, z up) of a server-spawned object at
 * (north, west, up) with orientation o (0 north, pi/2 west) and uniform scale.
 */
export function worldFromSpawnModel(x: number, y: number, z: number, north: number, west: number, up: number, orientation: number, scale: number): Vec3 {
	const c = Math.cos(orientation), s = Math.sin(orientation);
	return [north + scale * (c * x - s * y), west + scale * (s * x + c * y), up + scale * z];
}

/** Heading of a placed model's +X axis in WoW convention: 0 north, pi/2 west, in [0, 2*pi). */
export function yawOfMatrix(m: ArrayLike<number>): number {
	const north = -m[2];
	const west = -m[0];
	const yaw = Math.atan2(west, north);
	return yaw < 0 ? yaw + 2 * Math.PI : yaw;
}

/**
 * Client terrain triangulation for one chunk: per cell, four triangles fanned around the centre
 * vertex (outer 9x9 + inner 8x8 MCVT samples). Returns world-space triangles with their cell index
 * and quadrant, skipping holes and cells whose triangles miss the box.
 */
export function terrainChunkTriangles(
	tileX: number, tileY: number, chunk: Pick<AdtChunk, 'indexX' | 'indexY' | 'heights' | 'holes'>, box: WorldBox,
): { a: Vec3; b: Vec3; c: Vec3; local: number }[] {
	const out: { a: Vec3; b: Vec3; c: Vec3; local: number }[] = [];
	const row0 = tileY * 128 + chunk.indexY * 8;
	const col0 = tileX * 128 + chunk.indexX * 8;
	const outer = (r: number, c: number): Vec3 => [northOfCellRow(row0 + r), westOfCellCol(col0 + c), chunk.heights[r * 17 + c]];
	for (let r = 0; r < 8; r++) {
		for (let c = 0; c < 8; c++) {
			if (chunk.holes[r] & (1 << c)) continue;
			const tl = outer(r, c), tr = outer(r, c + 1), bl = outer(r + 1, c), br = outer(r + 1, c + 1);
			const centre: Vec3 = [(tl[0] + bl[0]) / 2, (tl[1] + tr[1]) / 2, chunk.heights[r * 17 + 9 + c]];
			const quads: [Vec3, Vec3][] = [[tl, tr], [tr, br], [br, bl], [bl, tl]];
			quads.forEach(([p, q], k) => {
				// Explorer/client winding (tl, centre, tr) is counter-clockwise from above in display space;
				// the display-to-world map is a proper rotation, so the order is kept.
				if (triangleTouchesBox(p, centre, q, box)) out.push({ a: p, b: centre, c: q, local: (r * 8 + c) * 4 + k });
			});
		}
	}
	return out;
}

/** World box of ADT tile (x, y). */
export function tileBox(x: number, y: number): WorldBox {
	return { northMin: MAP_ORIGIN - (y + 1) * TILE_SIZE, northMax: MAP_ORIGIN - y * TILE_SIZE, westMin: MAP_ORIGIN - (x + 1) * TILE_SIZE, westMax: MAP_ORIGIN - x * TILE_SIZE };
}

/** Tiles (x, y) whose area overlaps the box. */
export function tilesForBox(box: WorldBox): [number, number][] {
	const xs = [Math.floor((MAP_ORIGIN - box.westMax) / TILE_SIZE), Math.floor((MAP_ORIGIN - box.westMin) / TILE_SIZE)];
	const ys = [Math.floor((MAP_ORIGIN - box.northMax) / TILE_SIZE), Math.floor((MAP_ORIGIN - box.northMin) / TILE_SIZE)];
	const tiles: [number, number][] = [];
	for (let y = Math.max(0, ys[0]); y <= Math.min(63, ys[1]); y++) for (let x = Math.max(0, xs[0]); x <= Math.min(63, xs[1]); x++) tiles.push([x, y]);
	return tiles;
}

/**
 * Tiles to read for an arena. A tile the WDT lacks is an error when it overlaps the geometry box and
 * is declared (listed) when it only overlaps the search margin. Every present tile needs root and obj0.
 */
export function planTiles(wdtTiles: readonly (WdtTile | null)[], searchBox: WorldBox, geometryBox: WorldBox): { present: [number, number][]; absentOutsideGeometry: [number, number][] } {
	const present: [number, number][] = [];
	const absentOutsideGeometry: [number, number][] = [];
	for (const [x, y] of tilesForBox(searchBox)) {
		const tile = wdtTiles[y * 64 + x];
		if (!tile) {
			if (boxesOverlap(tileBox(x, y), geometryBox)) throw new Error(`Missing ADT tile ${x}_${y} overlaps the arena geometry box; refusing a partial arena`);
			absentOutsideGeometry.push([x, y]);
			continue;
		}
		if (!tile.files.root) throw new Error(`ADT tile ${x}_${y} has no root file`);
		if (!tile.files.obj0) throw new Error(`ADT tile ${x}_${y} has no obj0 file; its placements would be missing`);
		present.push([x, y]);
	}
	return { present, absentOutsideGeometry };
}

/**
 * Liquid tiles of a WMO group (MLIQ) in world space: tiles whose low nibble is not 0xF (vmangos
 * wmo.cpp and the Explorer parser use the same rule). Tile (i, j) spans WMO x in [x0 + i*c, x0 + (i+1)*c]
 * and y in [y0 + j*c, y0 + (j+1)*c] with c = 1600/3/128; its corner heights come from the vertex grid.
 */
export function wmoLiquidTiles(liquid: Pick<WmoLiquid, 'xVerts' | 'yVerts' | 'xTiles' | 'yTiles' | 'position' | 'heights' | 'tiles'>, matrix: ArrayLike<number>):
	{ tile: [number, number]; box: WorldBox; upMin: number; upMax: number; liquidNibble: number }[] {
	if (liquid.tiles.length !== liquid.xTiles * liquid.yTiles || liquid.heights.length !== liquid.xVerts * liquid.yVerts || liquid.xVerts !== liquid.xTiles + 1 || liquid.yVerts !== liquid.yTiles + 1) {
		throw new Error(`MLIQ grid is inconsistent: ${liquid.xVerts}x${liquid.yVerts} vertices, ${liquid.xTiles}x${liquid.yTiles} tiles, ${liquid.tiles.length} tile bytes`);
	}
	const out: { tile: [number, number]; box: WorldBox; upMin: number; upMax: number; liquidNibble: number }[] = [];
	for (let j = 0; j < liquid.yTiles; j++) {
		for (let i = 0; i < liquid.xTiles; i++) {
			const nibble = liquid.tiles[j * liquid.xTiles + i] & 0x0f;
			if (nibble === 0x0f) continue;
			const corners = [[i, j], [i + 1, j], [i, j + 1], [i + 1, j + 1]].map(([vi, vj]) => worldFromPlacement(transformPoint(matrix,
				liquid.position[0] + vi * WMO_LIQUID_CELL, liquid.position[1] + vj * WMO_LIQUID_CELL, liquid.heights[vj * liquid.xVerts + vi])));
			out.push({
				tile: [i, j], liquidNibble: nibble,
				box: { northMin: Math.min(...corners.map((c) => c[0])), northMax: Math.max(...corners.map((c) => c[0])), westMin: Math.min(...corners.map((c) => c[1])), westMax: Math.max(...corners.map((c) => c[1])) },
				upMin: Math.min(...corners.map((c) => c[2])), upMax: Math.max(...corners.map((c) => c[2])),
			});
		}
	}
	return out;
}

/** Coverage cells (local row, col) that a world box overlaps, edges inclusive, clipped to the grid. */
export function markedCells(box: WorldBox, layout: { row0: number; col0: number; rows: number; cols: number }): [number, number][] {
	const r0 = Math.max(cellRowOfNorth(box.northMax) - layout.row0, 0);
	const r1 = Math.min(cellRowOfNorth(box.northMin) - layout.row0, layout.rows - 1);
	const c0 = Math.max(cellColOfWest(box.westMax) - layout.col0, 0);
	const c1 = Math.min(cellColOfWest(box.westMin) - layout.col0, layout.cols - 1);
	const out: [number, number][] = [];
	for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) out.push([r, c]);
	return out;
}

/** Canonical JSON: sorted object keys, no whitespace; rejects non-finite numbers. */
export function canonicalJson(value: unknown): string {
	if (value === null || typeof value === 'boolean' || typeof value === 'string') return JSON.stringify(value);
	if (typeof value === 'number') {
		if (!Number.isFinite(value)) throw new Error('Non-finite number in canonical JSON');
		return JSON.stringify(Object.is(value, -0) ? 0 : value);
	}
	if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
	if (typeof value === 'object') {
		const entries = Object.entries(value as Record<string, unknown>).filter(([, v]) => v !== undefined).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0));
		return `{${entries.map(([k, v]) => `${JSON.stringify(k)}:${canonicalJson(v)}`).join(',')}}`;
	}
	throw new Error(`Unsupported value in canonical JSON: ${typeof value}`);
}

export function sha256(bytes: Uint8Array | string): string {
	return createHash('sha256').update(bytes).digest('hex');
}

export interface EncounterRules {
	/** vmangos ThreatRadius config default 50 yd from the combat start point (emulator-sourced). */
	threat_radius_yd: number;
	/** vmangos Creature::IsOutOfThreatArea: 12 s since the last leash extension (emulator-sourced). */
	leash_extension_timer_s: number;
	/** Pursuit speed used for the leash envelope: player run speed 7 yd/s (vmangos baseMoveSpeed MOVE_RUN). */
	runner_speed_yd_s: number;
	/** Lowest player level the arena is used for; the aggro radius grows as the player level falls. */
	player_level_min: number;
}

export interface ArenaSpec {
	spec_version: string;
	arena_id: string;
	description: string;
	product: string;
	map_id: number;
	map_directory: string;
	wdt_fdid: number;
	anchor_world: { north: number; west: number; up: number };
	bounds_local: { x_min: number; x_max: number; y_min: number; y_max: number };
	margin_yd: number;
	tile_search_margin_yd: number;
	/** Creature spawns are exported within the bounds grown by this margin (neighbours that can join a fight). */
	spawn_margin_yd: number;
	encounter_rules: EncounterRules;
	/** Opt-in, at most 0.35: server objects tilted up to this angle are placed yaw-only (serverObjectYaw). Absent: refused. */
	server_object_max_tilt_rad?: number;
	/** Opt-in: server objects left out of the geometry, each with a reason (listed in server_gameobjects_without_model); every guid must be in the geometry box. */
	server_object_skip?: { guid: number; reason: string }[];
}

export function validateSpec(spec: ArenaSpec): void {
	if (spec.spec_version !== SPEC_FORMAT) throw new Error(`Unsupported arena spec version ${spec.spec_version}`);
	if (!/^[a-z0-9][a-z0-9-]{2,63}$/.test(spec.arena_id)) throw new Error('arena_id must be lowercase letters, digits and dashes');
	const numbers = [spec.map_id, spec.wdt_fdid, spec.anchor_world.north, spec.anchor_world.west, spec.anchor_world.up,
		spec.bounds_local.x_min, spec.bounds_local.x_max, spec.bounds_local.y_min, spec.bounds_local.y_max, spec.margin_yd, spec.tile_search_margin_yd];
	if (!numbers.every(Number.isFinite)) throw new Error('Arena spec numbers must be finite');
	if (!(spec.bounds_local.x_min < spec.bounds_local.x_max && spec.bounds_local.y_min < spec.bounds_local.y_max)) throw new Error('Arena bounds are empty or reversed');
	if (spec.margin_yd < 0 || spec.tile_search_margin_yd < spec.margin_yd) throw new Error('Margins must satisfy 0 <= margin <= tile search margin');
	if (!(Number.isFinite(spec.spawn_margin_yd) && spec.spawn_margin_yd >= 0)) throw new Error('spawn_margin_yd must be finite and >= 0');
	const tilt = spec.server_object_max_tilt_rad;
	if (tilt !== undefined && !(typeof tilt === 'number' && tilt > 0 && tilt <= 0.35)) throw new Error('server_object_max_tilt_rad must be a number in (0, 0.35] when present');
	const skip = spec.server_object_skip;
	if (skip !== undefined && !(Array.isArray(skip) && skip.length > 0
		&& skip.every((s) => Number.isInteger(s?.guid) && s.guid > 0 && typeof s.reason === 'string' && s.reason.trim() !== '')
		&& new Set(skip.map((s) => s.guid)).size === skip.length)) throw new Error('server_object_skip must be a non-empty list of distinct {guid, reason} when present');
	const rules = spec.encounter_rules;
	if (!rules || ![rules.threat_radius_yd, rules.leash_extension_timer_s, rules.runner_speed_yd_s].every((v) => Number.isFinite(v) && v > 0)
		|| !(Number.isInteger(rules.player_level_min) && rules.player_level_min >= 1)) throw new Error('encounter_rules must hold positive finite radii, timer and speed and an integer player_level_min >= 1');
}

export function worldBoxOf(spec: ArenaSpec): WorldBox {
	const a = spec.anchor_world;
	const b = spec.bounds_local;
	return { northMin: a.north + b.x_min, northMax: a.north + b.x_max, westMin: a.west + b.y_min, westMax: a.west + b.y_max };
}

// ---------------------------------------------------------------------------------------------
// Export run (needs the game install). Everything below reads local files only.

interface FileRecord { fdid: number; role: string; bytes: number; sha256: string; md5: string; ckey_verified: boolean; encrypted_sections?: string[] }

interface Source { key: string; kind: string; [field: string]: unknown }

class MeshBuilder {
	readonly tris: number[] = [];
	readonly flags: number[] = [];
	readonly source: number[] = [];
	readonly local: number[] = [];
	constructor(private readonly anchor: Vec3) {}

	add(a: Vec3, b: Vec3, c: Vec3, flags: number, source: number, local: number): void {
		for (const p of [a, b, c]) {
			if (!p.every(Number.isFinite)) throw new Error(`Non-finite vertex in source ${source}`);
			this.tris.push(...localFromWorld(p, this.anchor));
		}
		this.flags.push(flags);
		this.source.push(source);
		this.local.push(local);
	}

	get count(): number {
		return this.flags.length;
	}
}

export interface Args { spec: string; out: string; wowDir: string; spawns: string; safeLocs: string | null; vmangosDb: string; report: string | null }

/**
 * Test-only fault injection: each hook replaces what the install provided. Any hook used is recorded in
 * the manifest's test_fault_injection list, which the Python loader rejects.
 */
export interface ExportHooks {
	wdt?: (wdt: Wdt) => Wdt;
	adt?: (tileX: number, tileY: number, adt: AdtRoot) => AdtRoot;
	file?: (fdid: number, role: string, bytes: Uint8Array) => Uint8Array;
}

function parseArgs(argv: string[]): Args {
	const value = (name: string) => {
		const i = argv.indexOf(name);
		return i >= 0 ? argv[i + 1] : undefined;
	};
	const spec = value('--spec');
	const out = value('--out');
	const vmangosDb = value('--vmangos-db');
	if (!spec || !out || !vmangosDb) throw new Error('Usage: exportArena.ts --spec <spec.json> --out <dir> --vmangos-db <mangos.sqlite> [--wow-dir <dir>] [--spawns <map.json>] [--safe-locs <csv>] [--report <json>]');
	return {
		spec, out, vmangosDb,
		wowDir: value('--wow-dir') ?? 'C:/Program Files (x86)/World of Warcraft',
		spawns: value('--spawns') ?? 'public/spawns/map0.json',
		safeLocs: value('--safe-locs') ?? null,
		report: value('--report') ?? null,
	};
}

function fileSha(path: string): string {
	return sha256(readFileSync(path));
}

function git(root: string, args: string[]): string {
	return execFileSync('git', ['-C', root, ...args], { encoding: 'utf8' }).trim();
}

/** Columns of a simple CSV line with double-quoted fields. */
function csvFields(line: string): string[] {
	const out: string[] = [];
	let field = '';
	let quoted = false;
	for (let i = 0; i < line.length; i++) {
		const ch = line[i];
		if (quoted) {
			if (ch === '"' && line[i + 1] === '"') { field += '"'; i++; }
			else if (ch === '"') quoted = false;
			else field += ch;
		} else if (ch === '"') quoted = true;
		else if (ch === ',') { out.push(field); field = ''; }
		else field += ch;
	}
	out.push(field);
	return out;
}

function m2Name(bytes: Uint8Array, md20: number): string {
	const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	const count = view.getUint32(md20 + 8, true);
	const offset = md20 + view.getUint32(md20 + 12, true);
	return new TextDecoder().decode(bytes.subarray(offset, offset + count)).replaceAll('\0', '');
}

/**
 * Raw per-group fields the Explorer parser does not keep: per-triangle flags and material from MOPY
 * (u8 flags, u8 material; 0xFF = collision only) or, in newer files, MPY2 (u16 flags, u16 material;
 * 0xFFFF = collision only), MODR doodad references and the group name offset. TrinityCore's
 * vmap4_extractor reads MPY2 with the same flag bits as MOPY.
 */
export function wmoGroupExtras(bytes: Uint8Array): { polyFormat: 'MOPY' | 'MPY2' | null; flags: Uint16Array; invisible: Uint8Array; modr: number[]; nameOffset: number } {
	const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	let mogp: { offset: number; size: number } | null = null;
	for (const c of chunks(bytes)) if (c.id === 'MOGP') mogp = c;
	if (!mogp) throw new Error('WMO group has no MOGP chunk');
	let polyFormat: 'MOPY' | 'MPY2' | null = null;
	let flags = new Uint16Array(0);
	let invisible = new Uint8Array(0);
	let modr: number[] = [];
	for (const c of chunks(bytes, mogp.offset + 68, mogp.offset + mogp.size)) {
		if (c.id === 'MOPY' || c.id === 'MPY2') {
			if (polyFormat) throw new Error('WMO group has more than one MOPY/MPY2 chunk');
			polyFormat = c.id;
			const stride = c.id === 'MOPY' ? 2 : 4;
			if (c.size % stride) throw new Error(`${c.id} size ${c.size} is not a multiple of ${stride}`);
			const n = c.size / stride;
			flags = new Uint16Array(n);
			invisible = new Uint8Array(n);
			for (let i = 0; i < n; i++) {
				const o = c.offset + i * stride;
				flags[i] = stride === 2 ? bytes[o] : view.getUint16(o, true);
				invisible[i] = (stride === 2 ? bytes[o + 1] === 0xff : view.getUint16(o + 2, true) === 0xffff) ? 1 : 0;
			}
		} else if (c.id === 'MODR') modr = Array.from({ length: c.size / 2 }, (_, i) => view.getUint16(c.offset + i * 2, true));
	}
	return { polyFormat, flags, invisible, modr, nameOffset: view.getUint32(mogp.offset, true) };
}

function wmoGroupNames(root: Uint8Array): Uint8Array {
	for (const c of chunks(root)) if (c.id === 'MOGN') return root.subarray(c.offset, c.offset + c.size);
	return new Uint8Array(0);
}

function cString(bytes: Uint8Array, offset: number): string {
	if (offset < 0 || offset >= bytes.length) return '';
	const end = bytes.indexOf(0, offset);
	return new TextDecoder().decode(bytes.subarray(offset, end < 0 ? bytes.length : end));
}

interface ModfExtra { min: Vec3; max: Vec3; flags: number; position: Vec3; rotation: Vec3 }

/** MODF rows with the fields parsePlacements drops (extents, flags), keyed by unique ID. */
function modfExtras(obj0: Uint8Array): Map<number, ModfExtra> {
	const view = new DataView(obj0.buffer, obj0.byteOffset, obj0.byteLength);
	const f = (o: number) => view.getFloat32(o, true);
	const out = new Map<number, ModfExtra>();
	for (const c of chunks(obj0)) {
		if (c.id !== 'MODF') continue;
		for (let o = c.offset; o + 64 <= c.offset + c.size; o += 64) {
			out.set(view.getUint32(o + 4, true), {
				position: [f(o + 8), f(o + 12), f(o + 16)], rotation: [f(o + 20), f(o + 24), f(o + 28)],
				min: [f(o + 32), f(o + 36), f(o + 40)], max: [f(o + 44), f(o + 48), f(o + 52)], flags: view.getUint16(o + 56, true),
			});
		}
	}
	return out;
}

/** MDDF/MODF rows that name models by path instead of file ID; this exporter cannot resolve them. */
function nameReferencedPlacements(obj0: Uint8Array): number {
	const view = new DataView(obj0.buffer, obj0.byteOffset, obj0.byteLength);
	let count = 0;
	for (const c of chunks(obj0)) {
		if (c.id === 'MDDF') for (let o = c.offset; o + 36 <= c.offset + c.size; o += 36) if (!(view.getUint16(o + 34, true) & 0x40)) count++;
		if (c.id === 'MODF') for (let o = c.offset; o + 64 <= c.offset + c.size; o += 64) if (!(view.getUint16(o + 56, true) & 0x8)) count++;
	}
	return count;
}

function boxOfPlacementExtents(min: Vec3, max: Vec3): WorldBox {
	const a = worldFromPlacement(min);
	const b = worldFromPlacement(max);
	return { northMin: Math.min(a[0], b[0]), northMax: Math.max(a[0], b[0]), westMin: Math.min(a[1], b[1]), westMax: Math.max(a[1], b[1]) };
}

type Row = Record<string, number | string | null>;

export interface ServerObjectPool { max_limit: number; description: string; mother_pool: number | null; objects: { guid: number; chance: number }[]; child_pools: { pool: number; chance: number }[] }

interface VmangosCreature { ids: number[]; respawn: [number, number]; group: Record<string, unknown> | null }
interface VmangosObject { state: number; type: number; data: number[]; pool: { entry: number; chance: number } | null }

/**
 * Reads what map0.json lacks from the VMaNGOS world database and checks every map0.json spawn against it
 * (same guid set in the box, entry, position within 0.006 yd, which is map0.json's 0.01 yd rounding).
 * Creature pools and linking are refused; a creature group is exported as data and refused when a member
 * has alternative entries (CreatureGroup::ChooseCreatureId would then apply).
 */
export function readVmangos(path: string, mapId: number, creatures: { guid: number; entry: number; north: number; west: number; up: number }[], creatureBox: WorldBox,
	objects: { guid: number; entry: number; north: number; west: number; up: number }[], objectBox: WorldBox) {
	const db = new DatabaseSync(path, { readOnly: true });
	try {
		const P = VMANGOS_PATCH;
		const one = (sql: string, ...params: (number | string)[]) => db.prepare(sql).get(...params) as Row | undefined;
		const all = (sql: string, ...params: (number | string)[]) => db.prepare(sql).all(...params) as Row[];
		const num = (v: unknown) => Number(v);
		const inBox = (sql: string, box: WorldBox, extra: string) => new Set(all(`${sql} where map = ? and position_x between ? and ? and position_y between ? and ? and patch_min <= ? and patch_max >= ? ${extra}`,
			mapId, box.northMin, box.northMax, box.westMin, box.westMax, P, P).map((r) => num(r.guid)));
		const dbCreatures = inBox('select guid from creature', creatureBox, 'and guid not in (select guid from game_event_creature where event > 0)');
		const mapCreatures = new Set(creatures.map((c) => c.guid));
		const missing = [...dbCreatures].filter((g) => !mapCreatures.has(g)), extra = [...mapCreatures].filter((g) => !dbCreatures.has(g));
		if (missing.length || extra.length) throw new Error(`map0.json and the vmangos DB disagree on the creatures in the spawn box: only in DB ${missing.slice(0, 5)}, only in map0.json ${extra.slice(0, 5)}`);
		const dbObjects = inBox('select guid from gameobject', objectBox, 'and guid not in (select guid from game_event_gameobject where event > 0)');
		const mapObjects = new Set(objects.map((o) => o.guid));
		const objMissing = [...dbObjects].filter((g) => !mapObjects.has(g)), objExtra = [...mapObjects].filter((g) => !dbObjects.has(g));
		if (objMissing.length || objExtra.length) throw new Error(`map0.json and the vmangos DB disagree on the game objects in the box: only in DB ${objMissing.slice(0, 5)}, only in map0.json ${objExtra.slice(0, 5)}`);
		const near = (r: Row, s: { north: number; west: number; up: number }) => Math.abs(num(r.position_x) - s.north) <= 0.006 && Math.abs(num(r.position_y) - s.west) <= 0.006 && Math.abs(num(r.position_z) - s.up) <= 0.006;

		const templates = new Map<number, Row>();
		const template = (entry: number): Row => {
			let row = templates.get(entry);
			if (!row) {
				row = one('select * from creature_template where entry = ? and patch <= ? order by patch desc limit 1', entry, P);
				if (!row) throw new Error(`creature_template ${entry} is missing for patch ${P}`);
				templates.set(entry, row);
			}
			return row;
		};
		const out = new Map<number, VmangosCreature>();
		for (const c of creatures) {
			const r = one('select * from creature where guid = ?', c.guid);
			if (!r || num(r.id) !== c.entry || !near(r, c)) throw new Error(`creature ${c.guid}: map0.json spawn (entry ${c.entry}) disagrees with the vmangos DB row`);
			// spawn_flags (disabled, dead, not visible, active, ...) change whether and how the creature spawns; none is modelled.
			if (num(r.spawn_flags)) throw new Error(`creature ${c.guid} has spawn_flags ${num(r.spawn_flags)}; spawn flags are not modelled, refusing`);
			if (one('select guid from pool_creature where guid = ? and patch_min <= ? and patch_max >= ?', c.guid, P, P)) throw new Error(`creature ${c.guid} is pooled; creature pools are unsupported`);
			if (one('select guid from creature_linking where guid = ? or master_guid = ?', c.guid, c.guid)) throw new Error(`creature ${c.guid} is linked (creature_linking); unsupported`);
			const ids = [r.id, r.id2, r.id3, r.id4, r.id5].map(num);
			for (const id of ids.filter(Boolean)) {
				template(id);
				if (one('select id from pool_creature_template where id = ? and patch_min <= ? and patch_max >= ?', id, P, P)) throw new Error(`creature entry ${id} is pooled by template; unsupported`);
			}
			let group: Record<string, unknown> | null = null;
			const membership = one('select leader_guid from creature_groups where member_guid = ? or leader_guid = ? limit 1', c.guid, c.guid);
			if (membership) {
				const leader = num(membership.leader_guid);
				const members = all('select member_guid, flags from creature_groups where leader_guid = ? order by member_guid', leader);
				for (const m of members) {
					const mr = one('select id2 from creature where guid = ?', num(m.member_guid));
					if (mr && num(mr.id2)) throw new Error(`creature group ${leader}: member ${m.member_guid} has alternative entries; CreatureGroup::ChooseCreatureId is unsupported`);
				}
				group = { leader_guid: leader, member_guids: members.map((m) => num(m.member_guid)), flags: num(members.find((m) => num(m.member_guid) === c.guid)?.flags ?? 0) };
			}
			out.set(c.guid, { ids, respawn: [num(r.spawntimesecsmin), num(r.spawntimesecsmax)], group });
		}

		const poolOf = (guid: number, entry: number) => {
			const byGuid = all('select pool_entry, chance from pool_gameobject where guid = ? and patch_min <= ? and patch_max >= ?', guid, P, P);
			const byEntry = all('select pool_entry, chance from pool_gameobject_template where id = ? and patch_min <= ? and patch_max >= ?', entry, P, P);
			if (byGuid.length + byEntry.length > 1) throw new Error(`game object ${guid} is listed in more than one pool`);
			const r = byGuid[0] ?? byEntry[0];
			return r ? { entry: num(r.pool_entry), chance: num(r.chance) } : null;
		};
		const objectsOut = new Map<number, VmangosObject>();
		for (const o of objects) {
			const r = one('select * from gameobject where guid = ?', o.guid);
			if (!r || num(r.id) !== o.entry || !near(r, o)) throw new Error(`game object ${o.guid}: map0.json spawn (entry ${o.entry}) disagrees with the vmangos DB row`);
			if (num(r.spawn_flags)) throw new Error(`game object ${o.guid} has spawn_flags ${num(r.spawn_flags)}; spawn flags are not modelled, refusing`);
			const tpl = one('select * from gameobject_template where entry = ? and patch <= ? order by patch desc limit 1', o.entry, P);
			if (!tpl) throw new Error(`gameobject_template ${o.entry} is missing for patch ${P}`);
			objectsOut.set(o.guid, { state: num(r.state), type: num(tpl.type), data: Array.from({ length: 24 }, (_, i) => num(tpl[`data${i}`])), pool: poolOf(o.guid, o.entry) });
		}

		const pools: Record<string, ServerObjectPool> = {};
		// Fail closed where vmangos would not spawn the pool the way pools.py samples it (vmangos 53d24d5c PoolManager.cpp):
		// CheckPoolAndChance (:653-667) skips rows with chance < 0 or > 100; PoolGroup::AddEntry (:161-167) treats a non-zero
		// chance as explicit only when the pool's max_limit is 1 (otherwise the member is equal-chanced); CheckPool (:171-182,
		// :1022-1029) drops POOL_FLAG_AUTO_SPAWN when there is no equal-chanced member and the explicit chances (summed as
		// float) are neither 0 nor 100; member flags (FLAG_SPAWN_ENABLE_IF_WORLD_POP_OVER_BLIZZLIKE, :36-43) and pool_template
		// flags (POOL_FLAG_MAXLIMIT_SCALING_LINEAR, :45-50) depend on the realm population. Each of these stops the export.
		const checkMembers = (entry: number, maxLimit: number, label: string, members: { id: number; chance: number; flags: number }[]) => {
			for (const m of members) {
				if (!(m.chance >= 0 && m.chance <= 100)) throw new Error(`pool ${entry}: ${label} ${m.id} has chance ${m.chance} outside [0, 100] (vmangos CheckPoolAndChance skips it); refusing`);
				if (m.flags) throw new Error(`pool ${entry}: ${label} ${m.id} has flags ${m.flags}; pool member flags are not modelled, refusing`);
			}
			const explicit = members.filter((m) => m.chance !== 0 && maxLimit === 1);
			let sum = 0;
			for (const m of explicit) sum = Math.fround(sum + Math.fround(m.chance));
			if (explicit.length === members.length && sum !== 100 && sum !== 0) {
				throw new Error(`pool ${entry} fails vmangos CheckPool: explicit chances sum to ${sum} with no equal-chance member, so it never auto-spawns; refusing`);
			}
		};
		const addPool = (entry: number): void => {
			if (pools[String(entry)]) return;
			const tpl = one('select max_limit, description, flags from pool_template where entry = ? and patch_min <= ? and patch_max >= ?', entry, P, P);
			if (!tpl) throw new Error(`pool_template ${entry} is missing for patch ${P}`);
			if (num(tpl.flags)) throw new Error(`pool_template ${entry} has flags ${num(tpl.flags)}; population-dependent pool flags are not modelled, refusing`);
			const objectRows = [
				...all(`select g.guid as guid, p.chance as chance, p.flags as flags from pool_gameobject p join gameobject g on g.guid = p.guid
					where p.pool_entry = ? and p.patch_min <= ? and p.patch_max >= ? and g.patch_min <= ? and g.patch_max >= ?`, entry, P, P, P, P),
				...all(`select g.guid as guid, p.chance as chance, p.flags as flags from pool_gameobject_template p join gameobject g on g.id = p.id
					where p.pool_entry = ? and p.patch_min <= ? and p.patch_max >= ? and g.patch_min <= ? and g.patch_max >= ?`, entry, P, P, P, P),
			].map((m) => ({ id: num(m.guid), chance: num(m.chance), flags: num(m.flags) })).sort((a, b) => a.id - b.id);
			checkMembers(entry, num(tpl.max_limit), 'member', objectRows);
			const objectsIn = objectRows.map((m) => ({ guid: m.id, chance: m.chance }));
			const mothers = all('select mother_pool, chance from pool_pool where pool_id = ?', entry);
			if (mothers.length > 1) throw new Error(`pool ${entry} has more than one mother pool`);
			const mother = mothers[0];
			const childRows = all('select pool_id, chance, flags from pool_pool where mother_pool = ? order by pool_id', entry).map((c) => ({ id: num(c.pool_id), chance: num(c.chance), flags: num(c.flags) }));
			checkMembers(entry, num(tpl.max_limit), 'child pool', childRows);
			const children = childRows.map((c) => ({ pool: c.id, chance: c.chance }));
			pools[String(entry)] = { max_limit: num(tpl.max_limit), description: String(tpl.description ?? ''), mother_pool: mother ? num(mother.mother_pool) : null, objects: objectsIn, child_pools: children };
			for (const c of children) addPool(c.pool);
			if (mother) addPool(num(mother.mother_pool));
		};
		for (const o of objectsOut.values()) if (o.pool) addPool(o.pool.entry);
		const sortedPools = Object.fromEntries(Object.keys(pools).map(Number).sort((a, b) => a - b).map((k) => [String(k), pools[String(k)]]));
		return { creatures: out, templates, objects: objectsOut, pools: sortedPools };
	} finally {
		db.close();
	}
}

export async function exportArena(args: Args, hooks: ExportHooks = {}): Promise<{ dir: string; digest: string; report: Record<string, unknown> }> {
	const started = performance.now();
	const timings: Record<string, number> = {};
	const lap = (label: string, since: number) => { timings[label] = Math.round(performance.now() - since); };
	const specBytes = readFileSync(args.spec);
	const spec = JSON.parse(specBytes.toString('utf8')) as ArenaSpec;
	validateSpec(spec);
	const finalDir = resolve(args.out, spec.arena_id);
	if (existsSync(finalDir)) throw new Error(`${finalDir} exists; arenas are immutable. Export under a new arena_id.`);

	const source = new NodeSource(args.wowDir);
	try {
		return await exportFromSource(args, hooks, spec, specBytes, source, started, timings, lap);
	} finally {
		await closeSource(source);
	}
}

/**
 * Closes every file handle NodeSource opened. NodeSource never closes them, and Node 26 turns a
 * handle collected while open into an ERR_INVALID_STATE crash after the export has finished.
 */
async function closeSource(source: NodeSource): Promise<void> {
	const files = (source as unknown as { files: Map<string, Promise<{ handle?: { close(): Promise<void> } }>> }).files;
	for (const pending of files.values()) {
		try {
			await (await pending).handle?.close();
		} catch {
			// A file that failed to open has no handle to close.
		}
	}
}

type SpawnFile = {
	source: string;
	creatures: { templates: Record<string, unknown[]>; spawns: number[][]; paths: Record<string, number[]>; templatePaths: Record<string, number[]> };
	objects: { templates: Record<string, unknown[]>; spawns: number[][] };
};

async function exportFromSource(
	args: Args, hooks: ExportHooks, spec: ArenaSpec, specBytes: Buffer, source: NodeSource, started: number,
	timings: Record<string, number>, lap: (label: string, since: number) => void,
): Promise<{ dir: string; digest: string; report: Record<string, unknown> }> {
	const anchor: Vec3 = [spec.anchor_world.north, spec.anchor_world.west, spec.anchor_world.up];
	const arenaBox = worldBoxOf(spec);
	const geometryBox = expandBox(arenaBox, spec.margin_yd);
	const searchBox = expandBox(arenaBox, spec.tile_search_margin_yd);
	const finalDir = resolve(args.out, spec.arena_id);
	let t = performance.now();
	const products = await CascStorage.listProducts(source);
	const product = products.find((p) => p.product === spec.product && p.active);
	if (!product) throw new Error(`Product ${spec.product} is not active in ${args.wowDir}`);
	const storage = await CascStorage.open(source, spec.product);
	lap('open_storage_ms', t);
	const root = (storage as unknown as { root: { getContentKeys(fdid: number): Uint8Array[] } }).root;
	const files: FileRecord[] = [];
	const read = async (fdid: number, role: string): Promise<Uint8Array> => {
		const bytes = await storage.readFile(fdid);
		const md5 = createHash('md5').update(bytes).digest('hex');
		const ckeys = root.getContentKeys(fdid).map((k) => Buffer.from(k).toString('hex'));
		const verified = ckeys.includes(md5);
		if (!verified) throw new Error(`File ${fdid} (${role}) content does not match its CASC content key`);
		if (!files.some((f) => f.fdid === fdid)) files.push({ fdid, role, bytes: bytes.length, sha256: sha256(bytes), md5, ckey_verified: verified });
		return hooks.file ? hooks.file(fdid, role, bytes) : bytes;
	};

	t = performance.now();
	const parsedWdt = parseWdt(await read(spec.wdt_fdid, 'wdt'));
	const wdt = hooks.wdt ? hooks.wdt(parsedWdt) : parsedWdt;
	const { present: tiles, absentOutsideGeometry } = planTiles(wdt.tiles, searchBox, geometryBox);
	const mesh = new MeshBuilder(anchor);
	const sources: Source[] = [];
	const addSource = (s: Source) => sources.push(s) - 1;

	// Coverage grid over the arena bounds, at terrain cell resolution.
	const row0 = cellRowOfNorth(arenaBox.northMax);
	const row1 = cellRowOfNorth(arenaBox.northMin);
	const col0 = cellColOfWest(arenaBox.westMax);
	const col1 = cellColOfWest(arenaBox.westMin);
	const rows = row1 - row0 + 1;
	const cols = col1 - col0 + 1;
	const layout = { row0, col0, rows, cols };
	const coverage = new Uint8Array(rows * cols);
	const mcnkCorners: Record<string, unknown>[] = [];
	const liquids: Record<string, unknown>[] = [];
	const objectFiles = new Map<number, { tile: [number, number]; bytes: Uint8Array }>();

	for (const [tx, ty] of tiles) {
		const tile = wdt.tiles[ty * 64 + tx]!;
		const parsedAdt = parseAdtRoot(await read(tile.files.root, `adt_root_${tx}_${ty}`));
		const adt = hooks.adt ? hooks.adt(tx, ty, parsedAdt) : parsedAdt;
		objectFiles.set(tile.files.obj0, { tile: [tx, ty], bytes: await read(tile.files.obj0, `adt_obj0_${tx}_${ty}`) });
		const hasMh2o = adt.chunkIds.includes('MH2O');
		for (const chunk of adt.chunks.slice().sort((a, b) => a.indexY - b.indexY || a.indexX - b.indexX)) {
			const cornerNorth = northOfCellRow(ty * 128 + chunk.indexY * 8);
			const cornerWest = westOfCellCol(tx * 128 + chunk.indexX * 8);
			const chunkBox: WorldBox = { northMin: cornerNorth - CHUNK_SIZE, northMax: cornerNorth, westMin: cornerWest - CHUNK_SIZE, westMax: cornerWest };
			if (!boxesOverlap(chunkBox, geometryBox)) continue;
			if ((chunk.flags & MCNK_LEGACY_LIQUID) && !hasMh2o) throw new Error(`ADT ${tx}_${ty} chunk ${chunk.indexX},${chunk.indexY} flags legacy MCLQ liquid (0x${chunk.flags.toString(16)}); unsupported`);
			const sourceIndex = addSource({
				key: `terrain/t${tx}_${ty}/c${chunk.indexX}_${chunk.indexY}`, kind: 'terrain_chunk', tile: [tx, ty], chunk: [chunk.indexX, chunk.indexY],
				area_id: chunk.areaId, mcnk_flags: chunk.flags, file_fdid: tile.files.root,
			});
			mcnkCorners.push({ source: sourceIndex, tile: [tx, ty], chunk: [chunk.indexX, chunk.indexY], header_position: [...chunk.position], computed_corner_world: { north: cornerNorth, west: cornerWest } });
			for (const tri of terrainChunkTriangles(tx, ty, chunk, geometryBox)) mesh.add(tri.a, tri.b, tri.c, TRI.TERRAIN | TRI.MOVEMENT_SOLID, sourceIndex, tri.local);
			for (let r = 0; r < 8; r++) {
				for (let c = 0; c < 8; c++) {
					const gr = ty * 128 + chunk.indexY * 8 + r - row0;
					const gc = tx * 128 + chunk.indexX * 8 + c - col0;
					if (gr < 0 || gr >= rows || gc < 0 || gc >= cols) continue;
					coverage[gr * cols + gc] = (coverage[gr * cols + gc] & COVER.WMO_LIQUID) | COVER.TERRAIN | (chunk.holes[r] & (1 << c) ? COVER.HOLE : 0);
				}
			}
		}
		for (const liquid of adt.liquids) {
			const cells = mh2oCells(liquid, tx, ty, layout);
			for (const cell of cells) coverage[cell] |= COVER.LIQUID;
			if (cells.length) liquids.push({ tile: [tx, ty], chunk: [liquid.chunk % 16, Math.floor(liquid.chunk / 16)], type: liquid.type, min_height: liquid.minHeight, max_height: liquid.maxHeight, cells_in_arena: cells.length });
		}
	}
	lap('terrain_ms', t);

	// Placements: unique IDs may be listed by several tiles; take each once, in uid order.
	t = performance.now();
	const m2Placements = new Map<number, { placement: Placement; tile: [number, number] }>();
	const wmoPlacements = new Map<number, { placement: Placement; tile: [number, number]; extra: ModfExtra }>();
	let nameReferenced = 0;
	for (const [, { tile, bytes }] of [...objectFiles].sort((a, b) => a[0] - b[0])) {
		nameReferenced += nameReferencedPlacements(bytes);
		const extras = modfExtras(bytes);
		for (const p of parsePlacements(bytes)) {
			if (p.kind === 'm2' && !m2Placements.has(p.uid)) m2Placements.set(p.uid, { placement: p, tile });
			if (p.kind === 'wmo' && !wmoPlacements.has(p.uid)) wmoPlacements.set(p.uid, { placement: p, tile, extra: extras.get(p.uid)! });
		}
	}
	if (nameReferenced) throw new Error(`${nameReferenced} placements reference models by name; cannot resolve them, refusing a partial arena`);

	const modelCache = new Map<number, { name: string; magic: 'MD21' | 'MD20' | null; collision: { positions: Float32Array; indices: Uint32Array } | null }>();
	const loadModel = async (fdid: number) => {
		let model = modelCache.get(fdid);
		if (!model) {
			const bytes = await read(fdid, 'm2');
			const magic = m2Magic(bytes);
			if (!magic) throw new Error(`Model ${fdid} is not an M2 file; unsupported`);
			const m2 = parseM2(bytes);
			model = { name: m2Name(bytes, m2.md20), magic, collision: m2.collision };
			modelCache.set(fdid, model);
		}
		return model;
	};
	const addModel = async (fdid: number, matrix: Mat4, base: Source, kindFlag: number): Promise<void> => {
		const model = await loadModel(fdid);
		const origin = worldFromPlacement(transformPoint(matrix, 0, 0, 0));
		const scale = Math.hypot(matrix[0], matrix[1], matrix[2]);
		const source: Source = { ...base, fdid, model_name: model.name, origin_world: { north: origin[0], west: origin[1], up: origin[2] }, yaw_world_rad: yawOfMatrix(matrix), scale, solid: model.collision !== null, collision_triangles: 0 };
		const kept: [Vec3, Vec3, Vec3, number][] = [];
		if (model.collision) {
			const { positions, indices } = model.collision;
			const world = (i: number) => worldFromPlacement(transformPoint(matrix, positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]));
			for (let k = 0; k < indices.length; k += 3) {
				const a = world(indices[k]), b = world(indices[k + 1]), c = world(indices[k + 2]);
				if (triangleTouchesBox(a, b, c, geometryBox)) kept.push([a, b, c, k / 3]);
			}
		}
		if (!kept.length && !pointInBox(origin, geometryBox)) return;
		source.collision_triangles = kept.length;
		const index = addSource(source);
		// vmangos extracts these doodads with MOD_M2 (model.cpp); unit LoS ignores them by default.
		for (const [a, b, c, local] of kept) mesh.add(a, b, c, kindFlag | TRI.MOVEMENT_SOLID | (base.vmangos_vmap ? TRI.VMAP_M2 : TRI.NOT_IN_VMANGOS_VMAP), index, local);
	};

	for (const [uid, { placement, tile }] of [...m2Placements].sort((a, b) => a[0] - b[0])) {
		// Quick reject on the origin; real models near the arena are far smaller than the search margin.
		const origin = worldFromPlacement(transformPoint(placement.matrix, 0, 0, 0));
		if (!pointInBox(origin, searchBox)) continue;
		await addModel(placement.fdid, placement.matrix, { key: `m2/uid${uid}`, kind: 'm2_adt', uid, tile, vmangos_vmap: true }, TRI.M2_ADT);
	}
	lap('m2_ms', t);

	t = performance.now();
	const skippedGroups: Record<string, unknown>[] = [];
	const wmoLiquids: Record<string, unknown>[] = [];
	for (const [uid, { placement, tile, extra }] of [...wmoPlacements].sort((a, b) => a[0] - b[0])) {
		if (!boxesOverlap(boxOfPlacementExtents(extra.min, extra.max), geometryBox)) continue;
		const rootBytes = await read(placement.fdid, 'wmo_root');
		const rootInfo = parseWmoRoot(rootBytes);
		const names = wmoGroupNames(rootBytes);
		const references = new Set<number>();
		for (let gi = 0; gi < rootInfo.groupFdids.length; gi++) {
			const groupFdid = rootInfo.groupFdids[gi];
			const bytes = await read(groupFdid, 'wmo_group');
			const group = parseWmoGroup(bytes);
			const { polyFormat, flags: polyFlags, invisible, modr, nameOffset } = wmoGroupExtras(bytes);
			const name = cString(names, nameOffset);
			const skip = vmapGroupSkipReason(group.flags, name);
			// Liquids first, for every group (also ones vmangos skips): the client still has the water.
			if ((group.flags & MOGP.LIQUID) && !group.liquid) throw new Error(`WMO ${placement.fdid} group ${gi} has liquid flag 0x1000 but no MLIQ chunk`);
			if (group.liquid) {
				const inArena = wmoLiquidTiles(group.liquid, placement.matrix).filter((lt) => boxesOverlap(lt.box, geometryBox));
				if (inArena.length) {
					const cells = new Set<number>();
					for (const lt of inArena) for (const [r, c] of markedCells(lt.box, layout)) cells.add(r * cols + c);
					for (const cell of cells) coverage[cell] |= COVER.WMO_LIQUID;
					wmoLiquids.push({
						source: `wmo/uid${uid}/g${gi}`, uid, group_index: gi, group_fdid: groupFdid, group_flags: group.flags, group_liquid_type: group.liquidType,
						root_flags: rootInfo.flags, vmangos_group_skipped: skip, tiles_in_geometry_box: inArena.length, cells_marked: cells.size,
						up_min: Math.min(...inArena.map((lt) => lt.upMin)), up_max: Math.max(...inArena.map((lt) => lt.upMax)),
						tile_nibbles: [...new Set(inArena.map((lt) => lt.liquidNibble))].sort((a, b) => a - b),
					});
				}
			}
			if (skip) {
				skippedGroups.push({ uid, group_index: gi, group_fdid: groupFdid, flags: group.flags, reason: skip });
				continue;
			}
			for (const r of modr) references.add(r);
			const triCount = group.indices.length / 3;
			if (!polyFormat || polyFlags.length !== triCount) throw new Error(`WMO ${placement.fdid} group ${gi}: ${polyFormat ?? 'no MOPY/MPY2'} has ${polyFlags.length} entries for ${triCount} triangles`);
			const world = (i: number) => worldFromPlacement(transformPoint(placement.matrix, group.positions[i * 3], group.positions[i * 3 + 1], group.positions[i * 3 + 2]));
			const kept: [Vec3, Vec3, Vec3, number, number][] = [];
			let renderOnly = 0;
			for (let k = 0; k < triCount; k++) {
				if (!vmapTriangleCollides(polyFlags[k])) { renderOnly++; continue; }
				const a = world(group.indices[k * 3]), b = world(group.indices[k * 3 + 1]), c = world(group.indices[k * 3 + 2]);
				if (triangleTouchesBox(a, b, c, geometryBox)) kept.push([a, b, c, k, invisible[k] ? TRI.WMO_INVISIBLE : 0]);
			}
			if (!kept.length) continue;
			const index = addSource({
				key: `wmo/uid${uid}/g${gi}`, kind: 'wmo_group', uid, tile, fdid: placement.fdid, group_fdid: groupFdid, group_index: gi, group_flags: group.flags,
				group_name: name, poly_format: polyFormat, collision_triangles: kept.length, non_collision_triangles_excluded: renderOnly,
				placement: { position: extra.position, rotation_deg: extra.rotation, doodad_set: placement.doodadSet, name_set: placement.nameSet ?? 0, modf_flags: extra.flags },
			});
			for (const [a, b, c, local, extraFlag] of kept) mesh.add(a, b, c, TRI.WMO | TRI.MOVEMENT_SOLID | TRI.VMAP_WMO | extraFlag, index, local);
		}
		// Doodads: vmangos extracts the placement's set, restricted to doodads referenced by extracted
		// groups (Doodad::ExtractSet). Set 0 and unreferenced doodads are kept but flagged NOT_IN_VMANGOS_VMAP.
		const sets = new Set([placement.doodadSet, 0]);
		for (const setIndex of [...sets].sort((a, b) => a - b)) {
			const set = rootInfo.doodadSets[setIndex];
			if (!set) continue;
			for (let di = set.start; di < set.start + set.count; di++) {
				const d = rootInfo.doodads[di];
				if (!d?.fdid) continue;
				const vmangos = doodadInVmangosVmap(setIndex, placement.doodadSet, references.has(di));
				const matrix = multiply(placement.matrix, compose(translation(...d.position), fromQuaternion(...d.rotation), scaling(d.scale)));
				await addModel(d.fdid, matrix, { key: `wmo/uid${uid}/set${setIndex}/d${di}`, kind: 'm2_wmo_doodad', parent_uid: uid, tile, doodad_set: setIndex, doodad_index: di, vmangos_vmap: vmangos }, TRI.M2_WMO_DOODAD);
			}
		}
	}
	lap('wmo_ms', t);
	const unknownCells = coverage.filter((v) => !(v & COVER.TERRAIN)).length;
	if (unknownCells) throw new Error(`${unknownCells} coverage cells inside the bounds have no terrain data; refusing an arena with unknown coverage`);

	// Independent binding evidence: server spawns (VMaNGOS), client WorldSafeLocs, client display models.
	t = performance.now();
	const anchorsBox = expandBox(arenaBox, spec.margin_yd);
	const spawnBox = expandBox(arenaBox, spec.spawn_margin_yd);
	const spawnsBytes = readFileSync(args.spawns);
	const spawnData = JSON.parse(spawnsBytes.toString('utf8')) as SpawnFile;
	const asPos = (s: number[]) => ({ guid: s[0], entry: s[1], north: s[2], west: s[3], up: s[4] });
	const vm = readVmangos(args.vmangosDb, spec.map_id, spawnData.creatures.spawns.filter((s) => pointInBox([s[2], s[3]], spawnBox)).map(asPos), spawnBox,
		spawnData.objects.spawns.filter((s) => pointInBox([s[2], s[3]], anchorsBox)).map(asPos), anchorsBox);
	const vmangosDbRecord = { path_basename: args.vmangosDb.split(/[\\/]/).pop(), sha256: fileSha(args.vmangosDb), patch: VMANGOS_PATCH,
		declared_source: 'VMaNGOS world database, sqlite release db_latest (db-sqlite.zip), the source tools/buildSpawns.ts names' };
	const creatures: Record<string, unknown>[] = [];
	const waypoints: Record<string, unknown>[] = [];
	const templatePathsSeen = new Set<number>();
	for (const s of spawnData.creatures.spawns) {
		const [guid, entry, north, west, up, orientation, movement = 0] = s;
		if (!pointInBox([north, west], anchorsBox)) continue;
		creatures.push({ guid, entry, name: spawnData.creatures.templates[String(entry)]?.[0] ?? null, north, west, up, orientation, movement });
		const own = movement === -1;
		if (!own && (movement !== -2 || templatePathsSeen.has(entry))) continue;
		if (!own) templatePathsSeen.add(entry);
		const path = own ? spawnData.creatures.paths[String(guid)] : spawnData.creatures.templatePaths[String(entry)];
		if (!path) throw new Error(`Creature ${guid}: movement ${movement} names a path the spawn file lacks`);
		for (let i = 0; i + 3 < path.length; i += 4) {
			if (pointInBox([path[i], path[i + 1]], anchorsBox)) waypoints.push({ guid, entry, point: i / 4, north: path[i], west: path[i + 1], up: path[i + 2], source: own ? 'guid' : 'template' });
		}
	}
	// GameObjectDisplayInfo has encrypted sections in this build; they read as zeros, so its content key
	// cannot match. The file is recorded with its encrypted key names instead of a verified hash.
	const displayRead = await storage.readFileWithStatus(GAMEOBJECT_DISPLAY_INFO_FDID, true);
	const displayMd5 = createHash('md5').update(displayRead.data).digest('hex');
	files.push({ fdid: GAMEOBJECT_DISPLAY_INFO_FDID, role: 'db2_gameobject_display_info', bytes: displayRead.data.length, sha256: sha256(displayRead.data), md5: displayMd5,
		ckey_verified: root.getContentKeys(GAMEOBJECT_DISPLAY_INFO_FDID).map((k) => Buffer.from(k).toString('hex')).includes(displayMd5), encrypted_sections: displayRead.encryptedKeys });
	const displayInfo = new Db2(displayRead.data);
	const objectsWithoutModel: Record<string, unknown>[] = [];
	const objectsSkipped: number[] = [];
	const skipReasons = new Map((spec.server_object_skip ?? []).map((s) => [s.guid, s.reason]));
	const gameobjects: Record<string, unknown>[] = [];
	let serverObjectTriangles = 0;
	for (const s of spawnData.objects.spawns) {
		const [guid, entry, north, west, up, orientation, r0, r1, r2, r3] = s;
		if (!pointInBox([north, west], anchorsBox)) continue;
		const template = spawnData.objects.templates[String(entry)] ?? [];
		const type = Number(template[1] ?? -1);
		const displayId = Number(template[2] ?? 0);
		const scale = Number(template[3] ?? 1);
		const modelFdid = displayInfo.has(displayId) ? displayInfo.getInt(displayId, 1) : null;
		const pooled = vm.objects.get(guid)!.pool;
		gameobjects.push({ guid, entry, name: template[0] ?? null, type, display_id: displayId, display_model_fdid: modelFdid, scale, north, west, up, orientation, rotation: [r0, r1, r2, r3],
			pool_entry: pooled ? pooled.entry : null });
		// Server-spawned objects the client draws and collides with. Yaw-only rotations are supported;
		// anything else fails the export instead of guessing.
		if (!pointInBox([north, west], geometryBox)) continue;
		const skipped = skipReasons.get(guid);
		if (skipped !== undefined) {
			objectsSkipped.push(guid);
			objectsWithoutModel.push({ guid, entry, name: template[0] ?? null, display_id: displayId, reason: `skipped by spec server_object_skip: ${skipped}` });
			continue;
		}
		if (!modelFdid) {
			objectsWithoutModel.push({ guid, entry, name: template[0] ?? null, display_id: displayId, reason: displayInfo.has(displayId) ? 'display has no model file' : 'display id not in GameObjectDisplayInfo' });
			continue;
		}
		const yaw = serverObjectYaw(guid, [r0, r1, r2, r3], orientation, spec.server_object_max_tilt_rad ?? 0);
		const tiltDropped = Math.abs(r0) >= 1e-3 || Math.abs(r1) >= 1e-3 ? serverObjectTilt([r0, r1, r2, r3]) : 0;
		const go = vm.objects.get(guid)!;
		if (go.type !== type) throw new Error(`Game object ${guid}: map0.json type ${type} != vmangos gameobject_template type ${go.type}`);
		const model = await loadModel(modelFdid);
		const los = serverObjectVmangosLos(go);
		const kept: [Vec3, Vec3, Vec3, number][] = [];
		if (model.collision) {
			const { positions, indices } = model.collision;
			const world = (i: number) => worldFromSpawnModel(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2], north, west, up, orientation, scale);
			for (let k = 0; k < indices.length; k += 3) kept.push([world(indices[k]), world(indices[k + 1]), world(indices[k + 2]), k / 3]);
		}
		const index = addSource({ key: `gameobject/guid${guid}`, kind: 'server_gameobject', guid, entry, gameobject_type: type, display_id: displayId, fdid: modelFdid, model_name: model.name,
			origin_world: { north, west, up }, yaw_world_rad: yaw, ...(tiltDropped ? { tilt_dropped_rad: tiltDropped } : {}), scale, solid: model.collision !== null, collision_triangles: kept.length,
			gameobject_state: go.state, vmangos_los: los, pool_entry: go.pool ? go.pool.entry : null, collision: go.pool ? 'conditional-pool' : 'permanent',
			provenance: 'VMaNGOS gameobject spawn (server data), model from client GameObjectDisplayInfo' });
		const objectFlags = TRI.SERVER_GAMEOBJECT | TRI.MOVEMENT_SOLID | (los.blocks_unit_los ? TRI.SERVER_LOS_BLOCKER : 0) | (go.pool ? TRI.CONDITIONAL_POOL : 0);
		for (const [a, b, c, local] of kept) mesh.add(a, b, c, objectFlags, index, local);
		serverObjectTriangles += kept.length;
	}
	const missingSkips = [...skipReasons.keys()].filter((g) => !objectsSkipped.includes(g));
	if (missingSkips.length) throw new Error(`server_object_skip names game objects outside the geometry box: ${missingSkips.join(', ')}`);
	const safeLocs: Record<string, unknown>[] = [];
	let safeLocsRecord: Record<string, unknown> | null = null;
	if (args.safeLocs) {
		const csv = readFileSync(args.safeLocs, 'utf8');
		safeLocsRecord = { path_basename: args.safeLocs.split(/[\\/]/).pop(), sha256: sha256(csv), declared_source: 'wago.tools WorldSafeLocs CSV, client build 1.13.2.31446' };
		const [header, ...lines] = csv.trim().split(/\r?\n/);
		const columns = csvFields(header);
		const col = (row: string[], name: string) => row[columns.indexOf(name)];
		for (const line of lines) {
			const row = csvFields(line);
			if (Number(col(row, 'Continent')) !== spec.map_id) continue;
			const north = Number(col(row, 'Loc_0')), west = Number(col(row, 'Loc_1')), up = Number(col(row, 'Loc_2'));
			if (pointInBox([north, west], anchorsBox)) safeLocs.push({ id: Number(col(row, 'ID')), name: col(row, 'AreaName_lang'), north, west, up, facing_deg: Number(col(row, 'Facing')) });
		}
	}
	lap('evidence_ms', t);

	// Encounter data for T3-T5: creature spawns, levels, faction reaction and movement (separate file).
	t = performance.now();
	const factionBytes = await read(FACTION_TEMPLATE_FDID, 'db2_faction_template');
	const factions = new Db2(factionBytes);
	const factionRow = (id: number): FactionTemplateRow | null => {
		if (!factions.has(id)) return null;
		const list = (field: number) => Array.from({ length: factions.arrayLength(field) }, (_, k) => factions.getInt(id, field, k) ?? 0);
		return { faction: factions.getInt(id, 0) ?? 0, group: factions.getInt(id, 2) ?? 0, friend: factions.getInt(id, 3) ?? 0, enemy: factions.getInt(id, 4) ?? 0, enemies: list(5), friends: list(6) };
	};
	// Player templates: 1 (Human, Alliance) and 2 (Orc, Horde).
	const alliance = factionRow(1), horde = factionRow(2);
	if (!alliance || !horde) throw new Error('FactionTemplate lacks the player templates 1 and 2');
	const encounter: Record<string, unknown>[] = [];
	for (const s of spawnData.creatures.spawns) {
		const [guid, entry, north, west, up, orientation, movement = 0] = s;
		if (!pointInBox([north, west], spawnBox)) continue;
		const tpl = spawnData.creatures.templates[String(entry)];
		if (!tpl) throw new Error(`Creature ${guid}: entry ${entry} has no template in the spawn file`);
		const [name, , levelMin, levelMax, , rank, , , , factionTemplate] = tpl as [string, string, number, number, number, number, number, unknown, unknown, number];
		const primary = vm.templates.get(entry)!;
		if (primary.name !== name || Number(primary.level_min) !== levelMin || Number(primary.level_max) !== levelMax || Number(primary.rank) !== rank || Number(primary.faction) !== factionTemplate) {
			throw new Error(`Creature ${guid}: map0.json template of entry ${entry} disagrees with the vmangos creature_template`);
		}
		const row = vm.creatures.get(guid)!;
		const alternatives: EncounterAlternative[] = entryAlternatives(guid, row.ids).map(({ entry: id, probability }) => {
			const c = vm.templates.get(id)!;
			return {
				entry: id, probability, name: String(c.name), subname: String(c.subname ?? ''), level_min: Number(c.level_min), level_max: Number(c.level_max), rank: Number(c.rank),
				creature_type: Number(c.type), npc_flags: Number(c.npc_flags), faction_template: Number(c.faction), reaction: reactionsFor(factionRow(Number(c.faction)), alliance, horde),
				detection_range_yd: Number(c.detection_range), call_for_help_range_yd: Number(c.call_for_help_range), leash_range_yd: Number(c.leash_range),
				speed_walk: Number(c.speed_walk), speed_run: Number(c.speed_run),
			};
		});
		const path = movement === -1 ? spawnData.creatures.paths[String(guid)] : movement === -2 ? spawnData.creatures.templatePaths[String(entry)] : null;
		encounter.push(creatureEncounter({ guid, world: [north, west, up], orientation, movement, path: path ?? null, alternatives, respawn: row.respawn, group: row.group }, spec));
	}
	encounter.sort((a, b) => (a.guid as number) - (b.guid as number));
	const spawnsFile = {
		format: SPAWNS_FORMAT, arena_id: spec.arena_id, bounds_local: spec.bounds_local, spawn_margin_yd: spec.spawn_margin_yd,
		provenance: {
			source: spawnData.source, file_basename: args.spawns.split(/[\\/]/).pop(), file_sha256: sha256(spawnsBytes),
			built_by: 'ClassicWowMapExplorer tools/buildSpawns.ts from the VMaNGOS world database: creature rows with patch_min <= 10 <= patch_max and no world-event spawns; positions rounded to 0.01 yd, orientation to 0.001 rad',
			faction_table: { fdid: FACTION_TEMPLATE_FDID, file: 'client FactionTemplate.db2 from the same install', rule: 'FactionTemplateEntry::IsHostileTo / IsFriendlyTo against player templates 1 (Human, Alliance) and 2 (Orc, Horde): Enemies[]/Friends[] naming the player faction first, then EnemyGroup & player group -> hostile, FriendGroup & player group or group & player FriendGroup -> friendly, else neutral; reputation-based reactions not modelled' },
			vmangos_db: { ...vmangosDbRecord, read: 'creature id..id5, spawntimesecsmin/max; creature_template (latest patch <= 10) of every alternative; creature_groups; gameobject state; gameobject_template type and data0-23; pool_gameobject, pool_gameobject_template, pool_template, pool_pool' },
			status: 'emulator-sourced (VMaNGOS world DB), not verified on the Forever build',
		},
		rules: {
			...spec.encounter_rules,
			entry_choice: 'vmangos CreatureData::ChooseCreatureId (Creature.cpp:63-76): uniform over the non-zero creature.id..id5 slots, counted up to the first zero; drawn again at each respawn',
			aggro_radius: 'vmangos Creature::GetAttackDistance (Creature.cpp:2193-2238): creature_template.detection_range (set in Creature.cpp:681; the 20 yd constructor default at Creature.cpp:261 is overwritten) minus (player level - creature level), the difference capped at 25 levels below (MAX_LEVEL_DIFF_FOR_AGGRO_RANGE, CreatureDefines.h:655), floored at min(detection_range, 5), 0 under 1 yd, times Rate.Creature.Aggro = 1; detect-range auras not modelled. aggro_radius_max_yd is the largest over the alternatives at level_max and player_level_min',
			threat_area: 'vmangos Creature::IsOutOfThreatArea (Creature.cpp:2796-2818): leash when neither creature nor victim is within max(1.5 * aggro, ThreatRadius = 50) of the combat start point and 12 s have passed since the last leash extension; being attacked (Unit.cpp:5987) or crowd-controlled (Creature.cpp:989) extends it. A non-zero creature_template.leash_range (leash_range_yd) is a hard chase limit from the combat start (Creature.cpp:992)',
			encounter_admissible: 'in bounds and edge_distance_yd >= engagement_radius_yd = roam_radius_yd + aggro_radius_max_yd + threat_area_radius_yd (roam = wander radius, or the farthest waypoint from the spawn); a superset of where creature and victim can be before a leash',
			leash_contained: 'in bounds and edge_distance_yd >= leash_envelope_yd = engagement_radius_yd + runner_speed_yd_s * leash_extension_timer_s; a lower bound: each leash extension adds up to another runner_speed * 12 s',
			sources: 'vmangos commit 53d24d5c; emulator-sourced, not verified on the Forever build',
		},
		creatures: encounter,
	};
	const spawnsBytesOut = Buffer.from(canonicalJson(spawnsFile) + '\n', 'utf8');
	lap('spawns_ms', t);

	// Assemble: arrays, geometry, evidence, spawns, manifest, lock. Written to a temporary directory, then renamed.
	const triangles = new Float64Array(mesh.tris);
	const arrays: Record<string, { data: Uint8Array; dtype: string; shape: number[] }> = {
		triangles: { data: new Uint8Array(triangles.buffer), dtype: '<f8', shape: [mesh.count, 3, 3] },
		tri_flags: { data: new Uint8Array(new Uint16Array(mesh.flags).buffer), dtype: '<u2', shape: [mesh.count] },
		tri_source: { data: new Uint8Array(new Uint32Array(mesh.source).buffer), dtype: '<u4', shape: [mesh.count] },
		tri_local: { data: new Uint8Array(new Uint32Array(mesh.local).buffer), dtype: '<u4', shape: [mesh.count] },
		coverage: { data: coverage, dtype: '|u1', shape: [rows, cols] },
	};
	const evidence = {
		format: EVIDENCE_FORMAT,
		note: 'Independent position sources for coordinate binding. Not used to build geometry.',
		sources: {
			vmangos_spawns: { path_basename: args.spawns.split(/[\\/]/).pop(), sha256: sha256(spawnsBytes), declared_source: spawnData.source, built_by: 'ClassicWowMapExplorer tools/buildSpawns.ts (VMaNGOS world DB, patch 10, rounded to 0.01 yd)' },
			vmangos_db: vmangosDbRecord,
			worldsafelocs: safeLocsRecord,
			gameobject_display_info: { fdid: GAMEOBJECT_DISPLAY_INFO_FDID, note: 'client GameObjectDisplayInfo.db2 from the same install; field 1 is the model file ID' },
		},
		creatures, waypoints, gameobjects, worldsafelocs: safeLocs, mcnk_corners: mcnkCorners,
	};
	const evidenceBytes = Buffer.from(canonicalJson(evidence) + '\n', 'utf8');
	const count = (bit: number) => mesh.flags.filter((f) => f & bit).length;
	const geometryPart = {
		map: { map_id: spec.map_id, directory: spec.map_directory, wdt_fdid: spec.wdt_fdid, instance_id: null, tiles: { present: tiles, absent_outside_geometry: absentOutsideGeometry } },
		frame: {
			id: FRAME_ID,
			units: 'yard',
			world_axes: 'WoW world position as used by the server and client: x north, y west, z up',
			anchor_world: spec.anchor_world,
			world_to_local: 'x = north - anchor.north; y = west - anchor.west; z = up - anchor.up',
			placement_to_world: 'north = 32*T - Pz; west = 32*T - Px; up = Py; T = 1600/3 (ADT placement space: Px east, Py up, Pz south)',
			explorer_display_to_world: 'Xe = 32*T - west + offsetX*T; Ye = up; Ze = 32*T - north + offsetY*T (ClassicWowMapExplorer src/viewer/regionData.ts worldFromWow)',
			tile_size_yd: TILE_SIZE,
			map_origin_yd: MAP_ORIGIN,
			source_precision_note: 'ADT and placement values are float32; near 9000 yd their spacing is about 0.001 yd, and composed float32 matrices near 26000 placement units round to about 0.002 yd.',
		},
		bounds_local: spec.bounds_local,
		geometry_margin_yd: spec.margin_yd,
		terrain: {
			cell_size_yd: CELL_SIZE,
			triangulation: 'client layout: per cell four triangles fanned around the inner (centre) vertex; outer 9x9 and inner 8x8 MCVT heights plus the MCNK base height',
			holes: 'cells flagged in MCNK holes (high-resolution 8x8 or low-resolution 4x4 bitmap) have no terrain triangles',
		},
		coverage: {
			array: 'coverage', row0_global: row0, col0_global: col0, rows, cols,
			row_axis: 'global terrain cell row; increases southward; row g spans north in [northOfRow(g+1), northOfRow(g)]',
			col_axis: 'global terrain cell column; increases eastward; column g spans west in [westOfCol(g+1), westOfCol(g)]',
			bits: { terrain: COVER.TERRAIN, hole: COVER.HOLE, liquid: COVER.LIQUID, wmo_liquid: COVER.WMO_LIQUID },
			zero_means: 'no terrain data: unknown (the exporter refuses such cells inside the bounds)',
			wmo_liquid_means: 'a WMO MLIQ liquid tile overlaps the cell at some height; queries treat the whole column as liquid',
			declared_unknown_cells: [],
		},
		liquids,
		wmo_liquids: wmoLiquids,
		triangle_flags: { ...TRI },
		rules_sources: {
			wmo_triangle_collision: 'vmangos 53d24d5c contrib/vmap_extractor/vmapextract/wmo.cpp WMOGroup::ConvertToVMAPGroupWmo; MPY2 handling as TrinityCore 73e86e6b src/tools/vmap4_extractor/wmo.cpp',
			wmo_group_skip: 'vmangos 53d24d5c contrib/vmap_extractor/vmapextract/wmo.cpp WMOGroup::ShouldSkip',
			wmo_doodads: 'vmangos 53d24d5c contrib/vmap_extractor/vmapextract/model.cpp Doodad::ExtractSet and vmapexport.cpp ExtractSingleWmo; doodads get MOD_M2 (model.cpp)',
			wmo_liquid: 'MLIQ per group; a tile has liquid unless its low nibble is 0xF (vmangos wmo.cpp, Explorer src/formats/wmo.ts)',
			line_of_sight: 'vmangos 53d24d5c Map.h isInLineOfSight(checkDynLos = true, ignoreM2Model = true); WorldModel::IntersectRay skips MOD_M2 models; vmaps contain no terrain',
			server_object_los: 'vmangos 53d24d5c: GameObjectModel::construct builds no model for a button/goober with losOK or a server-only generic/trap/spell focus; GameObject::UpdateCollisionState enables it only in GO_STATE_READY (chests while loot is ready); .m2 models get MOD_M2 unless GameObjectInfo::CanAlwaysBreakLoS (door, generic), and unit LoS ignores MOD_M2. Per source: vmangos_los',
			server_object_pools: 'vmangos 53d24d5c PoolManager.cpp: a pool spawns at most max_limit members (explicit chances rolled first, else uniform among equal-chance members); a mother pool (pool_pool) spawns child pools the same way. Pooled objects carry CONDITIONAL_POOL and collision conditional-pool; server_object_pools holds every pool they belong to with all members (also outside the arena), their mother pools and the mothers\' other children',
			server_object_orientation: 'unverified: the exporter places server-object collision with Rz(orientation) as vmangos GameObjectModel does (GameObjectModel.cpp:104; the vmap extractor keeps raw M2 axes), but at that yaw 13 of 20 Northshire Wooden Benches intersect client WMO walls and at yaw + 180 deg none do (critic probe bench_yaw_v2.json); which convention the client uses is open',
			m2_collision: 'M2 MD20 collision triangles/vertices at header offsets 0xd8/0xe0 (ClassicWowMapExplorer docs/prop-collision.md)',
		},
		sources,
		skipped_wmo_groups: skippedGroups,
		server_gameobjects_without_model: objectsWithoutModel,
		server_object_pools: vm.pools,
		counts: {
			triangles: mesh.count,
			terrain_triangles: count(TRI.TERRAIN),
			wmo_triangles: count(TRI.WMO),
			m2_adt_triangles: count(TRI.M2_ADT),
			m2_wmo_doodad_triangles: count(TRI.M2_WMO_DOODAD),
			server_gameobject_triangles: serverObjectTriangles,
			vmap_wmo_triangles: count(TRI.VMAP_WMO),
			vmap_m2_triangles: count(TRI.VMAP_M2),
			not_in_vmangos_vmap_triangles: count(TRI.NOT_IN_VMANGOS_VMAP),
			server_los_blocker_triangles: count(TRI.SERVER_LOS_BLOCKER),
			conditional_pool_triangles: count(TRI.CONDITIONAL_POOL),
			sources: sources.length,
			coverage_cells: rows * cols,
			coverage_unknown_cells: coverage.filter((v) => v === 0).length,
			coverage_hole_cells: coverage.filter((v) => v & COVER.HOLE).length,
			coverage_liquid_cells: coverage.filter((v) => v & COVER.LIQUID).length,
			coverage_wmo_liquid_cells: coverage.filter((v) => v & COVER.WMO_LIQUID).length,
		},
		arrays: Object.fromEntries(Object.entries(arrays).map(([name, a]) => [name, { file: `${name}.bin`, dtype: a.dtype, shape: a.shape, bytes: a.data.byteLength, sha256: sha256(a.data) }])),
	};
	const geometryBytes = Buffer.from(canonicalJson({ format: GEOMETRY_FORMAT, ...geometryPart }) + '\n', 'utf8');

	const explorerRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
	const toolPath = fileURLToPath(import.meta.url);
	const importedSources = ['src/casc/storage.ts', 'src/formats/chunks.ts', 'src/formats/adt.ts', 'src/formats/wdt.ts', 'src/formats/wmo.ts', 'src/formats/m2.ts',
		'src/formats/db2.ts', 'src/formats/mh2o.ts', 'src/explorer/objects.ts', 'src/explorer/mat4.ts', 'tools/nodeSource.ts'];
	const tileFiles = files.filter((f) => f.role.startsWith('adt_') || f.role === 'wdt');
	const manifest = {
		format: ARENA_FORMAT,
		arena_id: spec.arena_id,
		description: spec.description,
		use_rights: 'Private local extraction from the operator\'s own installed client for offline training, authorized by the operator. Game-derived geometry: keep in ignored folders; do not commit or redistribute.',
		spec: { sha256: sha256(specBytes), value: spec },
		build: {
			product: product.product, version: product.version, branch: product.branch, build_key: product.buildKey, cdn_key: product.cdnKey,
			build_name: storage.stats.buildName, build_info_sha256: fileSha(join(args.wowDir, '.build.info')),
		},
		...geometryPart,
		triangle_flag_meaning: {
			MOVEMENT_SOLID: 'terrain plus WMO collision triangles (vmangos MOPY rule) plus M2 collision meshes plus server object collision (pooled objects only while spawned, see CONDITIONAL_POOL)',
			VMAP_WMO: 'WMO collision in the vmangos static vmap; blocks unit LoS (no MOD_M2)',
			VMAP_M2: 'M2 collision (ADT doodads and the WMO placement doodad set referenced by extracted groups) in the vmangos static vmap with MOD_M2; ignored by unit LoS with the default ignoreM2Model = true',
			NOT_IN_VMANGOS_VMAP: 'WMO doodad collision that the vmangos vmap extractor omits: set 0 when the placement uses another set, or a doodad no extracted group references (MODR)',
			WMO_INVISIBLE: 'collision-only WMO triangles (MOPY material 0xFF or MPY2 material 0xFFFF)',
			SERVER_GAMEOBJECT: 'collision meshes of VMaNGOS game object spawns drawn from client models; server data, not client map geometry',
			SERVER_LOS_BLOCKER: 'server game object whose vmangos dynamic model exists, is enabled at its spawn state and blocks LoS with checkDynLos and ignoreM2Model (door or generic type)',
			CONDITIONAL_POOL: 'pooled server object (vmangos pool_gameobject / pool_gameobject_template): its triangles exist only while vmangos has it spawned; queries ignore them unless the caller names the object as spawned',
		},
		geometry: { file: 'geometry.json', format: GEOMETRY_FORMAT, sha256: sha256(geometryBytes) },
		evidence: { file: 'evidence.json', format: EVIDENCE_FORMAT, sha256: sha256(evidenceBytes) },
		spawns: { file: 'spawns.json', format: SPAWNS_FORMAT, sha256: sha256(spawnsBytesOut) },
		test_fault_injection: Object.keys(hooks).filter((k) => hooks[k as keyof ExportHooks]).sort(),
		source_files: { tile_files: tileFiles, model_files: files.filter((f) => !tileFiles.includes(f)).sort((a, b) => a.fdid - b.fdid) },
		exporter: {
			tool: 'ClassicWowMapExplorer/tools/exportArena.ts',
			tool_sha256: fileSha(toolPath),
			explorer_head: git(explorerRoot, ['rev-parse', 'HEAD']),
			explorer_status_porcelain: git(explorerRoot, ['status', '--porcelain']).split(/\r?\n/).filter(Boolean),
			imported_source_sha256: Object.fromEntries(importedSources.map((p) => [p, fileSha(join(explorerRoot, p))])),
			node: process.version,
		},
	};
	const manifestBytes = Buffer.from(canonicalJson(manifest) + '\n', 'utf8');
	const digest = sha256(manifestBytes);
	const geometryDigest = sha256(geometryBytes);
	const lock = { format: LOCK_FORMAT, arena_id: spec.arena_id, manifest_sha256: digest };

	const tempDir = resolve(args.out, `.${spec.arena_id}.partial-${process.pid}`);
	rmSync(tempDir, { recursive: true, force: true });
	mkdirSync(tempDir, { recursive: true });
	for (const [name, a] of Object.entries(arrays)) writeFileSync(join(tempDir, `${name}.bin`), a.data);
	writeFileSync(join(tempDir, 'geometry.json'), geometryBytes);
	writeFileSync(join(tempDir, 'evidence.json'), evidenceBytes);
	writeFileSync(join(tempDir, 'spawns.json'), spawnsBytesOut);
	writeFileSync(join(tempDir, 'manifest.json'), manifestBytes);
	writeFileSync(join(tempDir, 'arena.lock.json'), canonicalJson(lock) + '\n');
	if (existsSync(finalDir)) throw new Error(`${finalDir} appeared during export; refusing to overwrite`);
	renameSync(tempDir, finalDir);

	timings.total_ms = Math.round(performance.now() - started);
	const report = { arena_id: spec.arena_id, dir: finalDir, digest, geometry_digest: geometryDigest, counts: geometryPart.counts, tiles: geometryPart.map.tiles,
		wmo_liquids: wmoLiquids.length, spawns: encounter.length, files_read: files.length, timings, build: manifest.build, test_fault_injection: manifest.test_fault_injection };
	if (args.report) writeFileSync(args.report, JSON.stringify(report, null, 2) + '\n');
	return { dir: finalDir, digest, report };
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
	const result = await exportArena(parseArgs(process.argv.slice(2)));
	console.log(JSON.stringify(result.report, null, 2));
}
