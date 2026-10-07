import { Box3, BufferAttribute, BufferGeometry, DoubleSide, Line3, Matrix3, Matrix4, Ray, Sphere, Vector3 } from 'three';
import { ExtendedTriangle, MeshBVH } from 'three-mesh-bvh';
import type { CollisionData } from '../explorer/collision';

export interface SolidHit {
	distance: number;
	normal: Vector3;
	ground: number;
}

/** One prop's solid mesh, shared by its placements. Queries use current placement matrices. */
export class PropCollision {
	private readonly geometry = new BufferGeometry();
	private readonly tree: MeshBVH;
	private readonly bounds = new Sphere();
	private readonly worldBounds = new Sphere();
	private readonly ray = new Ray();
	private readonly inverse = new Matrix4();
	private readonly normalMatrix = new Matrix3();
	private readonly direction = new Vector3();
	private readonly point = new Vector3();
	private readonly segment = new Line3();
	private readonly localSegment = new Line3();
	private readonly sweptBounds = new Box3();
	private readonly triangle = new ExtendedTriangle();
	private readonly trianglePoint = new Vector3();
	private readonly segmentPoint = new Vector3();
	private readonly center = new Vector3();

	constructor(data: CollisionData) {
		this.geometry.setAttribute('position', new BufferAttribute(data.positions, 3));
		this.geometry.setIndex(new BufferAttribute(data.indices, 1));
		this.tree = MeshBVH.deserialize(data.bvh, this.geometry, { setIndex: false });
		this.tree.getBoundingBox(new Box3()).getBoundingSphere(this.bounds);
	}

	near(matrix: Matrix4, at: Vector3, margin: number): boolean {
		this.worldBounds.copy(this.bounds).applyMatrix4(matrix);
		const reach = this.worldBounds.radius + margin;
		return this.worldBounds.center.distanceToSquared(at) <= reach * reach;
	}

	cast(matrix: Matrix4, from: Vector3, direction: Vector3, far: number): SolidHit | null {
		this.worldBounds.copy(this.bounds).applyMatrix4(matrix);
		this.ray.set(from, direction);
		const closest = this.ray.closestPointToPoint(this.worldBounds.center, this.point);
		if (closest.distanceToSquared(this.worldBounds.center) > this.worldBounds.radius ** 2 ||
			from.distanceTo(this.worldBounds.center) > far + this.worldBounds.radius || matrix.determinant() === 0) return null;
		this.inverse.copy(matrix).invert();
		this.ray.origin.copy(from).applyMatrix4(this.inverse);
		const e = this.inverse.elements;
		this.direction.set(
			e[0] * direction.x + e[4] * direction.y + e[8] * direction.z,
			e[1] * direction.x + e[5] * direction.y + e[9] * direction.z,
			e[2] * direction.x + e[6] * direction.y + e[10] * direction.z,
		);
		const scale = this.direction.length();
		this.ray.direction.copy(this.direction).divideScalar(scale);
		const hit = this.tree.raycastFirst(this.ray, DoubleSide, 0, far * scale);
		if (!hit?.face) return null;
		const distance = hit.point.applyMatrix4(matrix).distanceTo(from);
		const normal = hit.face.normal.clone().applyNormalMatrix(this.normalMatrix.getNormalMatrix(matrix));
		if (normal.dot(direction) > 0) normal.negate();
		return { distance, normal, ground: 0 };
	}

	/** Continuous sphere contact, including thin fences and obstacles offset from the centre ray. */
	sweep(matrix: Matrix4, from: Vector3, move: Vector3, radius: number): SolidHit | null {
		const length = move.length();
		if (length < 1e-8 || !this.near(matrix, from, length + radius) || matrix.determinant() === 0) return null;
		this.inverse.copy(matrix).invert();
		this.segment.set(from, this.point.copy(from).add(move));
		this.localSegment.start.copy(this.segment.start).applyMatrix4(this.inverse);
		this.localSegment.end.copy(this.segment.end).applyMatrix4(this.inverse);
		this.sweptBounds.makeEmpty().expandByPoint(this.localSegment.start).expandByPoint(this.localSegment.end)
			.expandByScalar(radius * this.inverse.getMaxScaleOnAxis());
		let best: SolidHit | null = null;
		this.tree.shapecast({
			intersectsBounds: (box) => box.intersectsBox(this.sweptBounds),
			intersectsTriangle: (local) => {
				const tri = this.triangle;
				tri.a.copy(local.a).applyMatrix4(matrix);
				tri.b.copy(local.b).applyMatrix4(matrix);
				tri.c.copy(local.c).applyMatrix4(matrix);
				tri.needsUpdate = true;
				if (tri.closestPointToSegment(this.segment, this.trianglePoint, this.segmentPoint) > radius) return false;
				let hi = this.segment.closestPointToPointParameter(this.segmentPoint, true);
				let lo = 0;
				tri.closestPointToPoint(from, this.trianglePoint);
				this.direction.copy(from).sub(this.trianglePoint);
				if (this.direction.length() <= radius + 1e-6) {
					if (this.direction.lengthSq() < 1e-12) this.direction.copy(move).negate();
					if (this.direction.dot(move) >= -1e-8) return false;
					hi = 0;
				} else {
					for (let i = 0; i < 24; i++) {
						const mid = (lo + hi) * 0.5;
						this.segment.at(mid, this.center);
						tri.closestPointToPoint(this.center, this.trianglePoint);
						if (this.center.distanceToSquared(this.trianglePoint) <= radius * radius) hi = mid;
						else lo = mid;
					}
					this.segment.at(hi, this.center);
					tri.closestPointToPoint(this.center, this.trianglePoint);
					this.direction.copy(this.center).sub(this.trianglePoint);
				}
				const distance = hi * length;
				if (!best || distance < best.distance) best = { distance, normal: this.direction.clone().normalize(), ground: 0 };
				return false;
			},
		});
		return best;
	}

	dispose(): void {
		this.geometry.dispose();
	}
}
