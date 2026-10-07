import { mkdirSync, writeFileSync } from 'node:fs';
import { Box3, Matrix4, Ray, Vector3 } from 'three';
import { CascStorage } from '../src/casc/storage';
import { buildCollision } from '../src/explorer/collision';
import { MapExplorer } from '../src/explorer/maps';
import { parsePlacements } from '../src/explorer/objects';
import { parseM2 } from '../src/formats/m2';
import { PropCollision } from '../src/viewer/propCollision';
import { NodeSource } from './nodeSource';

const [wowDir = 'C:/Program Files (x86)/World of Warcraft', product = 'wow_classic_beta', tileX = '32', tileY = '48'] = process.argv.slice(2);
const [tx,ty]=[Number(tileX),Number(tileY)];
if (![tx,ty].every(n=>Number.isInteger(n)&&n>=0&&n<64)) throw new Error('Tile coordinates must be integers 0..63');
const storage=await CascStorage.open(new NodeSource(wowDir),product);
const maps=new MapExplorer(storage);
const tile=(await maps.wdt(775971)).tiles[ty*64+tx];
if (!tile?.files.obj0) throw new Error('Tile has no object file');
const placements=parsePlacements(await storage.readFile(tile.files.obj0));
const unique=new Map(placements.filter(p=>p.kind==='m2').map(p=>[p.fdid,p]));
const models: {fdid:number;name:string;vertices:number;triangles:number;rays:number}[]=[];
const errors: {fdid:number;error:string}[]=[];
const mismatches: {fdid:number;axis:number;expected:number|null;actual:number|null}[]=[];
let rays=0;
for (const [fdid,placement] of unique) {
	try {
		const bytes=await storage.readFile(fdid);
		const m2=parseM2(bytes);
		const view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
		const nameCount=view.getUint32(m2.md20+8,true);
		const nameOffset=m2.md20+view.getUint32(m2.md20+12,true);
		const name=new TextDecoder().decode(bytes.subarray(nameOffset,nameOffset+nameCount)).replaceAll('\0','');
		const data=buildCollision(m2.collision);
		const row={fdid,name,vertices:data?data.positions.length/3:0,triangles:data?data.indices.length/3:0,rays:0};
		models.push(row);
		if (!data) continue;
		const collider=new PropCollision(data);
		const matrix=new Matrix4().fromArray(placement.matrix);
		const vertices=Array.from({length:data.positions.length/3},(_,i)=>new Vector3().fromArray(data.positions,i*3).applyMatrix4(matrix));
		const bounds=new Box3().setFromPoints(vertices);
		const center=bounds.getCenter(new Vector3());
		const reach=bounds.getSize(new Vector3()).length()+2;
		for (const [axis,direction] of [new Vector3(1,0,0),new Vector3(-1,0,0),new Vector3(0,1,0),new Vector3(0,-1,0),new Vector3(0,0,1),new Vector3(0,0,-1)].entries()) {
			const from=center.clone().addScaledVector(direction,-reach);
			const ray=new Ray(from,direction);
			let expected:number|null=null;
			for (let i=0;i<data.indices.length;i+=3) {
				const point=ray.intersectTriangle(vertices[data.indices[i]],vertices[data.indices[i+1]],vertices[data.indices[i+2]],false,new Vector3());
				if (point) expected=Math.min(expected??Infinity,point.distanceTo(from));
			}
			const actual=collider.cast(matrix,from,direction,reach*2)?.distance??null;
			if ((actual===null)!==(expected===null)||(actual!==null&&expected!==null&&Math.abs(actual-expected)>1e-4)) mismatches.push({fdid,axis,expected,actual});
			row.rays++;rays++;
		}
		collider.dispose();
	} catch (error) {
		errors.push({fdid,error:error instanceof Error?error.message:String(error)});
	}
}
const report={schemaVersion:1,build:storage.stats,scope:'Offline installed-asset collision checks; no live-game or movement-parity acceptance',tile:{mapId:0,x:tx,y:ty},
	summary:{models:models.length,solid:models.filter(m=>m.triangles>0).length,nonSolid:models.filter(m=>m.triangles===0).length,rays,errors:errors.length,mismatches:mismatches.length},models,errors,mismatches};
mkdirSync('.cache',{recursive:true});
const reportPath=`.cache/collision-${product}-${tx}-${ty}.json`;
writeFileSync(reportPath,JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({build:report.build.version,reportPath,...report.summary}));
process.exit(errors.length||mismatches.length?1:0);
