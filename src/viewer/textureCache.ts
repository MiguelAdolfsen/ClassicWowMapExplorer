import type * as THREE from 'three';
import type { AsyncStorageApi } from '../worker/protocol';
import { createTexture } from './textures';

interface Entry {
	texture: Promise<THREE.Texture | null>;
	refs: number;
}

/**
 * Shared, reference-counted terrain layer textures. Neighbouring tiles mostly use the same
 * tileset textures, so each is read and uploaded once while any tile needs it.
 */
export class TextureCache {
	private readonly entries = new Map<number, Entry>();

	constructor(
		private readonly storage: Pick<AsyncStorageApi, 'loadTextures'>,
		private readonly compressed: boolean,
		private readonly anisotropy: number,
	) {}

	get size(): number {
		return this.entries.size;
	}

	/** Takes a reference to each texture, loading the missing ones in one worker call. */
	async acquire(fdids: number[]): Promise<Map<number, THREE.Texture | null>> {
		const missing = fdids.filter((id) => !this.entries.has(id));
		if (missing.length) {
			const batch = this.storage.loadTextures(missing, this.compressed);
			missing.forEach((fdid, i) => {
				const texture = batch.then((loaded) => {
					const data = loaded[i].texture;
					return data ? createTexture(data, this.anisotropy, true) : null;
				});
				this.entries.set(fdid, { texture, refs: 0 });
			});
		}
		for (const id of fdids) this.entries.get(id)!.refs++;
		const textures = await Promise.all(fdids.map((id) => this.entries.get(id)!.texture));
		return new Map(fdids.map((id, i) => [id, textures[i]]));
	}

	release(fdids: number[]): void {
		for (const id of fdids) {
			const entry = this.entries.get(id);
			if (!entry || --entry.refs > 0) continue;
			this.entries.delete(id);
			void entry.texture.then((t) => t?.dispose());
		}
	}
}
