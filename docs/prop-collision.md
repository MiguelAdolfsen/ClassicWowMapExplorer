# Prop collision

Date: 2026-10-07. The fork extends upstream revision `7099e75d0ae4f49229247f008433cf4f016390da`.

M2 props have dedicated collision vertices and triangle indices in the MD20 header, including
when that header is inside an MD21 chunk. These are separate from skin/render triangles.
The parser reads the collision index array at header offset `0xd8` and vertex array at `0xe0`,
validates their bounds and values, and retains the original model axes. Format references:
[TrinityCore model header](https://github.com/TrinityCore/TrinityCore/blob/master/src/tools/vmap4_extractor/modelheaders.h)
and [wow.export collision reader](https://github.com/Kruithne/wow.export/blob/master/src/js/3D/loaders/M2Loader.js).
Those references describe the layout; the installed-build probe provides the separate compatibility evidence.

## Behavior

The storage worker builds a BVH for dedicated M2 collision geometry. The viewer shares it across
the model's placed copies and applies each copy's current transformation when querying it.
Prop queries use loaded placement matrices independently of rendering visibility. Removing a
placement removes its collision immediately; disposing the unused model releases its geometry.

Floor, wall, camera and static sight queries include solid props. Movement uses continuous sphere
contacts to account for radius and prevent a long step from tunnelling through a thin prop.
Walking retains the existing samples at different body heights and the existing step-height behavior.
Render leaves, effects and skins are not substituted for absent collision geometry. Creatures are
not admitted as prop obstacles. The slope overlay still colors terrain and WMO render surfaces.

The explorer simulates its own character and reads local game assets. This change has no connection
to a running game process and sends no game input.

## Verification

Run from the repository root:

```powershell
npm ci --ignore-scripts --no-audit --no-fund
npm test
npm run build
npm run probe:collision -- "C:/Program Files (x86)/World of Warcraft" wow_classic_beta 32 48
```

The nine automated checks cover raw and chunked M2 headers, missing solids, invalid counts/indices/
vertices/offsets, chunk isolation, two-sided rays and range, changed/scaled/rotated placements,
thin off-center fences, movement away from a touching wall, and ObjectManager collision under
render culling and tile removal. A regression check confirms visual-only props remain non-solid
and WMO ray and sight behavior still works. Generated fixtures contain no game assets.

On 2026-10-07 the probe read installed build `1.60.1.70170`, product `wow_classic_beta`, and Azeroth
tile 32_48. It decoded 76 M2 models: 51 with collision triangles and 25 without them. All 306
transformed BVH ray comparisons matched brute-force world-space triangle intersections within
the declared `0.0001` asset-unit tolerance, with zero loading/parsing errors. The comparison
checks BVH queries and placements, not independent game collision truth. Its receipt is
`.cache/collision-wow_classic_beta-32-48.json`; it is local, ignored, and can be regenerated.

Representative solid files included ElwynnTreeMid01, ElwynnWoodFence01, ElwynnStoneFence,
ElwynnCliffRock02 and Crate01. ElwynnBush09 contained no collision mesh and remained non-solid.

A real browser loaded the same Forever product and the fork's worker/viewer. An 80-step walking
check aimed eight asset units across an ElwynnTreeMid01 placement, file ID 189932, using the real
WalkControls movement method and loaded world. The uncorrected straight path approached the
solid triangles to `0.00966` units. Collision produced a sideways slide with maximum lateral
displacement `0.93947` units and minimum measured triangle clearance `0.40306` units for the
tested body samples of radius `0.4`. The tree case demonstrates collision response, not every
prop or walk animation. The scene had 13,575 loaded prop-collision placements. The diagnostic
loop took about one second including triangle-distance measurements; this is not an FPS benchmark.

The browser also emitted flowing-water shader warnings and a gear-model loading warning.
The focused collision checks did not resolve or classify those warnings.

## Remaining limits

- Exact Forever walking, jump, step, slope and collision parity is unverified in the live client.
- Prop collision shapes are static. Animated doors and shape changes are not tracked.
- Props without authored collision triangles remain non-solid. Missing or malformed assets are
  reported through existing model-loading warnings; the viewer is not a certified navigation oracle.
- Collision requires loaded placements. This change does not make unloaded areas safe to route through.
- The existing bilinear terrain-height approximation and WMO collision behavior remain.
- Broad performance acceptance and continuous full-body capsule coverage remain unverified.
- No client geometry, recordings or personal game data are committed.
