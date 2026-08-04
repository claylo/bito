# Changelog

All notable changes to this project will be documented in this file.

This file is generated from commit history. Behavior changes that need action
on upgrade are written by hand in [docs/UPGRADING.md](docs/UPGRADING.md) —
put them there, not here, or the next release overwrites them.

## [Unreleased]

### Features

- *(cli)* Exit 1 when checks find issues, 2 when bito cannot run
- *(update)* Notify about new releases on info and doctor

### Bug Fixes

- *(observability)* Use platform-correct log directories and atomic writes (#3) by @claylo in [#3](https://github.com/claylo/bito/pull/3)
- *(logging)* Hand log file handling to librebar
- *(deps)* Clear the rmcp and anyhow security advisories

### Documentation

- Fix brew tap location by @claylo
- *(record)* Add librebar 0.4.0 adoption design spec
- *(record)* Fold librebar's error handling patterns into the plan
- *(handoff)* Record tasks 4 through 8

### Refactor

- *(config)* Drop bito-lint config names and simplify discovery (#2) by @claylo in [#2](https://github.com/claylo/bito/pull/2)
- *(config)* Load configuration through librebar
- *(cli)* Adopt librebar CommonArgs
- *(doctor)* Build on librebar's diagnostics framework

### Miscellaneous Tasks

- Workflow refactor by @claylo
- Update justfile recipes and rust toolchain to 1.97.1
- *(lint)* Clear clippy errors from the 1.97.1 toolchain bump

### Build

- *(deps)* Add librebar 0.4 to both crates
- *(deps)* Move to librebar 0.5.0 and refresh the plan
- *(xtask)* Generate artifacts through librebar



**Full Changelog**: https://github.com/claylo/bito/commits/HEAD

## [1.0.0] - 2026-03-25

### Features

- [**breaking**] Rename bito-lint to bito, adopt ah-ah-ah for token counting (#1) by @claylo in [#1](https://github.com/claylo/bito/pull/1)

### Bug Fixes

- *(ci)* Bump cargo-binstall action to v1.17.8 by @claylo
- Rename npm packages and fix npm publishing by @claylo




**Full Changelog**: [v0.3.0...v1.0.0](https://github.com/claylo/bito/compare/v0.3.0...v1.0.0)


## [0.3.0] - 2026-03-06

### Features

- Add config discovery, region suppression filtering, and custom content entries (#v0.3) (#20) by @claylo

### Documentation

- Update sample configs and README for v0.2.0 (#19) by @claylo




**Full Changelog**: [v0.2.0...v0.3.0](https://github.com/claylo/bito/compare/v0.2.0...v0.3.0)


## [0.2.0] - 2026-03-02

### Features

- Add path-based lint rules, inline suppressions, and lint CLI/MCP (#18) by @claylo




**Full Changelog**: [v0.1.7...v0.2.0](https://github.com/claylo/bito/compare/v0.1.7...v0.2.0)


## [0.1.7] - 2026-02-28

### Miscellaneous Tasks

- Formatting and workflow fixes (#16) by @claylo
- Update claylo-rs and rmcp deps (#17) by @claylo




**Full Changelog**: [v0.1.6...v0.1.7](https://github.com/claylo/bito/compare/v0.1.6...v0.1.7)


## [0.1.6] - 2026-02-28

### Features

- *(core)* Implement pluggable tokenizer backends (#10) by @claylo
- *(tokens)* Table-aware Claude tokenizer + vocab update to 38,360 (#12) by @claylo

### Bug Fixes

- Address code review findings 1-5 (#13) by @claylo

### Miscellaneous Tasks

- *(template)* Update claylo-rs (#11) by @claylo
- *(template)* Update claylo-rs to v1.0.0-beta.6 (#14) by @claylo
- Fix cd release workflow (#15) by @claylo




**Full Changelog**: [v0.1.5...v0.1.6](https://github.com/claylo/bito/compare/v0.1.5...v0.1.6)


## [0.1.3] - 2026-02-09

### Bug Fixes

- *(build)* Remove lld rustflags that broke linux-arm64-gnu cross-compilation (#4) by @claylo

### Miscellaneous Tasks

- *(cliff)* Fix changelog template (#6) by @claylo




**Full Changelog**: [v0.1.2...v0.1.3](https://github.com/claylo/bito/compare/v0.1.2...v0.1.3)


## [0.1.2] - 2026-02-09

### Bug Fixes

- *(build)* Help linux-arm64-gnu find the right ld (#3) by @claylo




**Full Changelog**: [v0.1.1...v0.1.2](https://github.com/claylo/bito/compare/v0.1.1...v0.1.2)


## [0.1.1] - 2026-02-09

### Bug Fixes

- *(build)* Make sure build targets are installed (#2) by @claylo




**Full Changelog**: [v0.1.0...v0.1.1](https://github.com/claylo/bito/compare/v0.1.0...v0.1.1)


## [0.1.0] - 2026-02-09

### Miscellaneous Tasks

- First commit by @claylo
- Cleanup (#1) by @claylo

### New Contributors
* @claylo made their first contribution





