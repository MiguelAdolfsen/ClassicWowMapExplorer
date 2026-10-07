/** Blending modes shared by M2 and WMO materials. */
export const Blend = {
	Opaque: 0,
	AlphaKey: 1,
	Alpha: 2,
	NoAlphaAdd: 3,
	Add: 4,
	Mod: 5,
	Mod2x: 6,
	BlendAdd: 7,
} as const;

export const M2_MATERIAL_UNLIT = 0x1;
export const M2_MATERIAL_UNFOGGED = 0x2;
export const M2_MATERIAL_TWO_SIDED = 0x4;
export const M2_MATERIAL_NO_DEPTH_WRITE = 0x10;

const TEXTURE_WRAP_X = 0x1;
const TEXTURE_WRAP_Y = 0x2;

export interface M2Texture {
	/** 0 = hard-coded file; other types are filled in at runtime (skins, capes, ...). */
	type: number;
	fdid: number;
	wrapX: boolean;
	wrapY: boolean;
}

export interface M2File {
	version: number;
	/** The whole file and the offset of its MD20 header, for reading animation tracks. */
	bytes: Uint8Array;
	md20: number;
	/** 48-byte vertex records. */
	vertices: Uint8Array;
	vertexCount: number;
	textures: M2Texture[];
	materials: { flags: number; blend: number }[];
	textureCombos: number[];
	/** Static opacity per transparency slot (first key of the first animation), 0-1. */
	transparency: number[];
	/** Static colour alpha per colour slot, 0-1. */
	colorAlpha: number[];
	/** Per texture-transform combo: steady UV scroll in texture units per second (scrolling fire, water). */
	uvScroll: ([number, number] | null)[];
	skinFdids: number[];
	bounds: { min: [number, number, number]; max: [number, number, number]; radius: number };
	collision: M2Collision | null;
}

export interface M2Collision {
	/** Dedicated solid geometry in the model's original z-up coordinates. */
	positions: Float32Array;
	indices: Uint32Array;
}

export interface M2Batch {
	/** Geoset ID of the batch's submesh (group * 100 + variant). */
	geoset: number;
	indexStart: number;
	indexCount: number;
	materialIndex: number;
	textureComboIndex: number;
	textureCount: number;
	transparencyIndex: number;
	/** Index into textureTransformCombos, or 0xffff for none. */
	uvAnimationIndex: number;
	colorIndex: number;
	priorityPlane: number;
	flags: number;
}

export interface M2Skin {
	/** Skin vertex -> model vertex. */
	vertexLookup: Uint16Array;
	/** Triangles, indexing vertexLookup. */
	indices: Uint16Array;
	batches: M2Batch[];
}

const decoder = new TextDecoder();

function arr(view: DataView, base: number, at: number): { count: number; offset: number } {
	return { count: view.getUint32(base + at, true), offset: view.getUint32(base + at + 4, true) };
}

/** Reads the first value of an M2Track's first animation, or null if the track is empty. */
function firstTrackValue(view: DataView, base: number, track: number, read: (at: number) => number): number | null {
	const values = arr(view, base, track + 12);
	if (values.count === 0) return null;
	const first = arr(view, base, values.offset);
	if (first.count === 0) return null;
	return read(base + first.offset);
}

/**
 * Parses a chunked (MD21) M2 model: static geometry, textures and materials. Animation is
 * ignored except for the static opacity of each batch.
 */
export function parseM2(bytes: Uint8Array): M2File {
	const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	let md20 = -1;
	let md20End = bytes.length;
	let skinFdids: number[] = [];
	let textureFdids: number[] = [];
	for (let p = 0; p + 8 <= bytes.length;) {
		const id = decoder.decode(bytes.subarray(p, p + 4));
		const size = view.getUint32(p + 4, true);
		const data = p + 8;
		if (id === 'MD21') {
			md20 = data;
			md20End = data + size;
		}
		else if (id === 'SFID') skinFdids = Array.from({ length: size / 4 }, (_, i) => view.getUint32(data + i * 4, true));
		else if (id === 'TXID') textureFdids = Array.from({ length: size / 4 }, (_, i) => view.getUint32(data + i * 4, true));
		p = data + size;
	}
	if (md20 < 0) {
		if (decoder.decode(bytes.subarray(0, 4)) !== 'MD20') throw new Error('Not an M2 model');
		md20 = 0;
	}
	const b = md20;
	const version = view.getUint32(b + 4, true);
	const vertices = arr(view, b, 0x3c);
	const skinCount = view.getUint32(b + 0x44, true);
	const colors = arr(view, b, 0x48);
	const textures = arr(view, b, 0x50);
	const weights = arr(view, b, 0x58);
	const materials = arr(view, b, 0x70);
	const textureCombos = arr(view, b, 0x80);
	const weightCombos = arr(view, b, 0x90);
	const f = (o: number) => view.getFloat32(b + o, true);

	const texList: M2Texture[] = [];
	for (let i = 0; i < textures.count; i++) {
		const o = b + textures.offset + i * 16;
		const flags = view.getUint32(o + 4, true);
		texList.push({ type: view.getUint32(o, true), fdid: textureFdids[i] ?? 0, wrapX: !!(flags & TEXTURE_WRAP_X), wrapY: !!(flags & TEXTURE_WRAP_Y) });
	}
	const matList = Array.from({ length: materials.count }, (_, i) => ({
		flags: view.getUint16(b + materials.offset + i * 4, true),
		blend: view.getUint16(b + materials.offset + i * 4 + 2, true),
	}));
	const combos = Array.from({ length: textureCombos.count }, (_, i) => view.getUint16(b + textureCombos.offset + i * 2, true));

	// Texture weights: M2Track<fixed16> (20 bytes each), looked up through the weight combos.
	const weightValues = Array.from({ length: weights.count }, (_, i) =>
		firstTrackValue(view, b, weights.offset + i * 20, (at) => view.getInt16(at, true) / 32767) ?? 1);
	const transparency = Array.from({ length: weightCombos.count }, (_, i) => weightValues[view.getUint16(b + weightCombos.offset + i * 2, true)] ?? 1);
	// Texture transforms (60 bytes: translation, rotation, scale tracks), looked up through the combos.
	// Only steady scrolling is kept: the translation's change from first to last key, per second.
	const transforms = arr(view, b, 0x60);
	const transformCombos = arr(view, b, 0x98);
	const scrollOf = (i: number): [number, number] | null => {
		if (i >= transforms.count) return null;
		const track = b + transforms.offset + i * 60;
		const times = arr(view, b, track - b + 4);
		const values = arr(view, b, track - b + 12);
		if (!times.count || !values.count) return null;
		const t = arr(view, b, times.offset);
		const v = arr(view, b, values.offset);
		const n = Math.min(t.count, v.count);
		if (n < 2) return null;
		const t0 = view.getUint32(b + t.offset, true), t1 = view.getUint32(b + t.offset + (n - 1) * 4, true);
		if (t1 <= t0) return null;
		const x = (k: number) => view.getFloat32(b + v.offset + k * 12, true);
		const y = (k: number) => view.getFloat32(b + v.offset + k * 12 + 4, true);
		const perSecond = 1000 / (t1 - t0);
		return [(x(n - 1) - x(0)) * perSecond, (y(n - 1) - y(0)) * perSecond];
	};
	const uvScroll = Array.from({ length: transformCombos.count }, (_, i) => scrollOf(view.getUint16(b + transformCombos.offset + i * 2, true)));

	// Colours: M2Color = color track (vec3) + alpha track (fixed16), 40 bytes.
	const colorAlpha = Array.from({ length: colors.count }, (_, i) =>
		firstTrackValue(view, b, colors.offset + i * 40 + 20, (at) => view.getInt16(at, true) / 32767) ?? 1);

	return {
		version,
		bytes,
		md20: b,
		vertices: bytes.subarray(b + vertices.offset, b + vertices.offset + vertices.count * 48),
		vertexCount: vertices.count,
		textures: texList,
		materials: matList,
		textureCombos: combos,
		transparency,
		colorAlpha,
		uvScroll,
		skinFdids: skinFdids.slice(0, Math.max(1, skinCount)),
		bounds: { min: [f(0xa0), f(0xa4), f(0xa8)], max: [f(0xac), f(0xb0), f(0xb4)], radius: f(0xb8) },
		collision: readCollision(view, b, md20End),
	};
}

function readCollision(view: DataView, base: number, end: number): M2Collision | null {
	if (base + 0xf0 > end || end > view.byteLength) throw new Error('Truncated M2 collision header');
	const triangles = arr(view, base, 0xd8);
	const vertices = arr(view, base, 0xe0);
	if (triangles.count === 0) return null;
	if (triangles.count % 3 || vertices.count === 0) throw new Error('Invalid M2 collision counts');
	const fits = (a: { count: number; offset: number }, stride: number) =>
		a.offset >= 0xf0 && base + a.offset + a.count * stride <= end;
	if (!fits(triangles, 2) || !fits(vertices, 12)) throw new Error('M2 collision array exceeds model data');
	const positions = new Float32Array(vertices.count * 3);
	for (let i = 0; i < positions.length; i++) {
		const value = view.getFloat32(base + vertices.offset + i * 4, true);
		if (!Number.isFinite(value)) throw new Error('Non-finite M2 collision vertex');
		positions[i] = value;
	}
	const indices = new Uint32Array(triangles.count);
	for (let i = 0; i < indices.length; i++) {
		const index = view.getUint16(base + triangles.offset + i * 2, true);
		if (index >= vertices.count) throw new Error('M2 collision index exceeds vertex count');
		indices[i] = index;
	}
	return { positions, indices };
}

export function parseSkin(bytes: Uint8Array): M2Skin {
	const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
	if (decoder.decode(bytes.subarray(0, 4)) !== 'SKIN') throw new Error('Not an M2 skin');
	const vertices = arr(view, 0, 0x04);
	const indices = arr(view, 0, 0x0c);
	const submeshes = arr(view, 0, 0x1c);
	const batches = arr(view, 0, 0x24);
	const u16s = (a: { count: number; offset: number }) => {
		const out = new Uint16Array(a.count);
		for (let i = 0; i < a.count; i++) out[i] = view.getUint16(a.offset + i * 2, true);
		return out;
	};
	const sections = Array.from({ length: submeshes.count }, (_, i) => {
		const o = submeshes.offset + i * 48;
		const level = view.getUint16(o + 2, true);
		return { geoset: view.getUint16(o, true), indexStart: view.getUint16(o + 8, true) + (level << 16), indexCount: view.getUint16(o + 10, true) };
	});
	const batchList: M2Batch[] = [];
	for (let i = 0; i < batches.count; i++) {
		const o = batches.offset + i * 24;
		const section = sections[view.getUint16(o + 4, true)];
		if (!section) continue;
		batchList.push({
			geoset: section.geoset,
			flags: view.getUint8(o),
			priorityPlane: view.getInt8(o + 1),
			indexStart: section.indexStart,
			indexCount: section.indexCount,
			colorIndex: view.getInt16(o + 8, true),
			materialIndex: view.getUint16(o + 10, true),
			textureCount: view.getUint16(o + 14, true),
			textureComboIndex: view.getUint16(o + 16, true),
			transparencyIndex: view.getUint16(o + 20, true),
			uvAnimationIndex: view.getUint16(o + 22, true),
		});
	}
	return { vertexLookup: u16s(vertices), indices: u16s(indices), batches: batchList };
}
