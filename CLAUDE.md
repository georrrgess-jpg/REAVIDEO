# REAVIDEO

- **Default editing style:** every edit, remake or recut in this repo uses the voidfx style. Load the `voidfx-edit` skill (`.claude/skills/voidfx-edit/SKILL.md`) first and follow its rules, workflow and QA checklist. Only deviate when the user asks; if they change a preference, update the skill's rules and its preference log.
- Reverse-engineering a new reference edit: use the rea-video tool (https://github.com/JuneYaooo/rea-video), then record anything new in `docs/style-breakdown.md` or the skill.
- Stored scene packs and reference edits: see `scenepacks/README.md`. Fetch them with `bash scenepacks/fetch.sh` when `media/` is empty, since cloud containers start fresh.
- Never commit footage, music or renders. `media/`, `edit/clips/` and `edit/out/` are gitignored.
