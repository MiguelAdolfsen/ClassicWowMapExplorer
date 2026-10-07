# Changelog

## 2026-10-07

- Added dedicated M2 prop collision parsing and worker-built BVHs so trees, fences, rocks,
  crates and other authored solid props participate in the explorer's floor, wall, camera
  and static sight queries. Kept collision geometry separate from rendering and tied it
  to loaded placements rather than draw-distance visibility.
- Added continuous radius sweeps for thin and off-center obstacles, migrated internal
  building-only query names to solid-geometry names, and retained existing WMO and terrain behavior.
- Added eight automated collision checks, an installed-asset probe, and validation documentation.
  Narrowed the storage types used by ObjectManager and TextureCache to their required methods.
- Validation: `npm test` passed eight checks; TypeScript checking passed; the installed Forever
  `1.60.1.70170` probe on tile 32_48 decoded 76 models and matched all 306 ray comparisons with
  zero errors. A browser walking check slid around a real tree while retaining at least 0.40306
  asset units of clearance for tested 0.4-radius body samples. Production build and final diff
  checks are pending at this entry's creation and will be recorded below after execution.
- Remaining: live-client movement parity, animated collision shapes, unloaded-area coverage,
  full-body capsule acceptance and broad performance checks. No game-control output was added.
- Final validation: `npm run build` passed TypeScript checking and the Vite production build;
  `git diff --check` passed. The earlier pending build/diff checks are now complete.
- Added a ninth regression check for visual-only props and WMO queries; `npm test` passed
  all nine checks. The production build and staged diff checks were rerun after this addition.
