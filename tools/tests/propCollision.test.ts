import assert from 'node:assert/strict';
import test from 'node:test';
import { Matrix4, Vector3 } from 'three';
import { buildCollision } from '../../src/explorer/collision';
import { parseM2 } from '../../src/formats/m2';
import { PropCollision } from '../../src/viewer/propCollision';
import { ObjectManager } from '../../src/viewer/objects';
import type { ModelData } from '../../src/explorer/objects';

const wall = {
	positions: new Float32Array([0,-10,-10, 0,10,-10, 0,10,10, 0,-10,10]),
	indices: new Uint32Array([0,1,2, 0,2,3]),
};
const identity = new Matrix4();
const x = new Vector3(1,0,0);

function collider(data = wall): PropCollision {
	const built = buildCollision(data);
	assert.ok(built);
	return new PropCollision(structuredClone(built));
}

function m2(chunked: boolean): Uint8Array {
	const base = chunked ? 8 : 0;
	const bytes = new Uint8Array(base + 0x166);
	const view = new DataView(bytes.buffer);
	bytes.set(new TextEncoder().encode(chunked ? 'MD21' : 'MD20'));
	if (chunked) {
		view.setUint32(4, bytes.length - base, true);
		bytes.set(new TextEncoder().encode('MD20'), base);
	}
	view.setUint32(base+4,264,true);
	view.setUint32(base+0xd8,3,true);
	view.setUint32(base+0xdc,0x160,true);
	view.setUint32(base+0xe0,3,true);
	view.setUint32(base+0xe4,0x130,true);
	[0,0,0, 0,1,0, 0,0,1].forEach((v,i)=>view.setFloat32(base+0x130+i*4,v,true));
	[0,1,2].forEach((v,i)=>view.setUint16(base+0x160+i*2,v,true));
	return bytes;
}

test('MD20 and MD21 read dedicated collision geometry with header-relative offsets', () => {
	for (const chunked of [false,true]) {
		const data = parseM2(m2(chunked)).collision;
		assert.ok(data);
		assert.deepEqual([...data.positions], [0,0,0,0,1,0,0,0,1]);
		assert.deepEqual([...data.indices], [0,1,2]);
	}
});

test('empty collision stays non-solid, and malformed collision never becomes an empty mesh', () => {
	const bytes = m2(false);
	const view = new DataView(bytes.buffer);
	view.setUint32(0xd8,0,true);
	assert.equal(parseM2(bytes).collision,null);
	view.setUint32(0xd8,2,true);
	assert.throws(()=>parseM2(bytes),/counts/);
	view.setUint32(0xd8,3,true);
	view.setUint16(0x160,3,true);
	assert.throws(()=>parseM2(bytes),/index/);
	view.setUint16(0x160,0,true);
	view.setFloat32(0x130,NaN,true);
	assert.throws(()=>parseM2(bytes),/Non-finite/);
	view.setFloat32(0x130,0,true);
	view.setUint32(0xe4,0x164,true);
	assert.throws(()=>parseM2(bytes),/exceeds/);
});

test('MD21 arrays cannot read into a following unrelated chunk', () => {
	const bytes = m2(true);
	new DataView(bytes.buffer).setUint32(4,0x140,true);
	assert.throws(()=>parseM2(bytes),/exceeds/);
});

test('collision rays hit both sides, respect range, and miss clear space', () => {
	const solid=collider();
	assert.equal(solid.cast(identity,new Vector3(-2,0,0),x,3)?.distance,2);
	assert.equal(solid.cast(identity,new Vector3(2,0,0),x.clone().negate(),3)?.distance,2);
	assert.equal(solid.cast(identity,new Vector3(-2,0,0),x,1),null);
	assert.equal(solid.cast(identity,new Vector3(-2,20,0),x,3),null);
	assert.equal(solid.near(identity,new Vector3(100,0,0),1),false);
	solid.dispose();
});

test('scaled, rotated and changed placements preserve world distances and floor normals', () => {
	const solid=collider();
	const matrix=new Matrix4().makeScale(2,3,0.5).setPosition(3,0,0);
	assert.equal(solid.cast(matrix,new Vector3(0,0,0),x,4)?.distance,3);
	matrix.setPosition(5,0,0);
	assert.equal(solid.cast(matrix,new Vector3(0,0,0),x,4),null);
	const floor=solid.cast(new Matrix4().makeRotationZ(Math.PI/2),new Vector3(0,3,0),new Vector3(0,-1,0),4);
	assert.ok(floor);
	assert.ok(Math.abs(floor.distance-3)<1e-6);
	assert.ok(floor.normal.y>0.999);
	assert.equal(solid.cast(new Matrix4().makeScale(0,1,1),new Vector3(-2,0,0),x,3),null);
	solid.dispose();
});

test('continuous radius catches a thin fence beside the centre ray and prevents tunnelling', () => {
	const solid=collider({positions:new Float32Array([0,-2,0.35,0,2,0.35,0,2,1,0,-2,1]),indices:wall.indices});
	const start=new Vector3(-2,0,0);
	assert.equal(solid.cast(identity,start,x,10),null);
	const contact=solid.sweep(identity,start,new Vector3(10,0,0),0.4);
	assert.ok(contact);
	assert.ok(Math.abs(contact.distance-1.80635083)<1e-5);
	assert.ok(contact.normal.x<0);
	assert.equal(solid.sweep(identity,start,new Vector3(-10,0,0),0.4),null);
	solid.dispose();
});

test('touching a wall allows tangential and outward movement', () => {
	const solid=collider();
	const start=new Vector3(-0.4,0,0);
	assert.equal(solid.sweep(identity,start,new Vector3(-1,0,0),0.4),null);
	assert.equal(solid.sweep(identity,start,new Vector3(0,0,1),0.4),null);
	assert.equal(solid.sweep(identity,start,new Vector3(1,0,0),0.4)?.distance,0);
	solid.dispose();
});

test('ObjectManager uses loaded prop solids despite render culling, and removes them with their tile', async () => {
	const model: ModelData={fdid:1,positions:wall.positions,normals:new Float32Array(12),uvs:new Float32Array(8),baked:null,indices:wall.indices,
		batches:[],radius:15,height:10,collision:buildCollision(wall)};
	const manager=new ObjectManager({
		loadTileObjects:async()=>[{kind:'m2',fdid:1,uid:1,matrix:new Float32Array(identity.elements),doodadSet:0}],
		loadModels:async()=>[model],loadTextures:async()=>[],
	},false,1,async()=>{});
	manager.setTileLevel('one',1,0,0,new Vector3(),'all');
	const camera=new Vector3(10000,10000,10000);
	await new Promise(resolve=>setImmediate(resolve));
	manager.update(1000,camera);
	await new Promise(resolve=>setImmediate(resolve));
	manager.update(2000,camera);
	assert.equal(manager.castSolid(new Vector3(-2,0,0),x,3)?.distance,2);
	assert.ok(manager.nearSolid(new Vector3(-2,0,0),1));
	assert.ok(manager.blocksSight(new Vector3(-2,0,0),new Vector3(2,0,0)));
	assert.equal(manager.sweep(new Vector3(-2,0,0),new Vector3(10,0,0),0.4).x,0);
	manager.setTileLevel('one',1,0,0,new Vector3(),'none');
	assert.equal(manager.castSolid(new Vector3(-2,0,0),x,3),null);
});

test('visual-only props stay non-solid while existing WMO floors and sight queries still work', async () => {
	const built=buildCollision(wall);
	assert.ok(built);
	const base:ModelData={fdid:2,positions:wall.positions,normals:new Float32Array(12),uvs:new Float32Array(8),baked:null,indices:wall.indices,radius:15,height:10,
		batches:[{start:0,count:6,order:0,material:{texture:0,blend:0,twoSided:true,unlit:true,unfogged:true,opacity:1,ground:7}}]};
	const building:ModelData={...base,fdid:3,bvh:{version:1,roots:built.bvh.roots,indirectBuffer:built.bvh.indirectBuffer}};
	const manager=new ObjectManager({
		loadTileObjects:async()=>[
			{kind:'m2',fdid:2,uid:2,matrix:new Float32Array(new Matrix4().makeTranslation(1,0,0).elements),doodadSet:0},
			{kind:'wmo',fdid:3,uid:3,matrix:new Float32Array(new Matrix4().makeTranslation(5,0,0).elements),doodadSet:0},
		],
		loadModels:async requests=>requests.map(r=>r.fdid===3?building:base),loadTextures:async()=>[],
	},false,1,async()=>{});
	manager.setTileLevel('two',1,0,0,new Vector3(),'all');
	await new Promise(resolve=>setImmediate(resolve));
	manager.update(1000,new Vector3());
	await new Promise(resolve=>setImmediate(resolve));
	manager.update(2000,new Vector3());
	assert.equal(manager.castSolid(new Vector3(),x,8)?.distance,5);
	assert.equal(manager.castSolid(new Vector3(),x,8)?.ground,7);
	assert.equal(manager.blocksSight(new Vector3(),new Vector3(2,0,0)),false);
	assert.equal(manager.blocksSight(new Vector3(),new Vector3(6,0,0)),true);
	manager.setTileLevel('two',1,0,0,new Vector3(),'none');
});
