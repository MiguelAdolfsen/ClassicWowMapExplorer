import { BufferAttribute, BufferGeometry } from 'three';
import { MeshBVH } from 'three-mesh-bvh';
import type { M2Collision } from '../formats/m2';

export type CollisionData = M2Collision & { bvh: ReturnType<typeof MeshBVH.serialize> };

export function buildCollision(data: M2Collision | null): CollisionData | null {
	if (!data) return null;
	const positions = data.positions.slice();
	const indices = data.indices.slice();
	const geometry = new BufferGeometry();
	geometry.setAttribute('position', new BufferAttribute(positions, 3));
	geometry.setIndex(new BufferAttribute(indices, 1));
	const bvh = MeshBVH.serialize(new MeshBVH(geometry, { indirect: true }), { cloneBuffers: false });
	geometry.dispose();
	return { positions, indices, bvh };
}
