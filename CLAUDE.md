# REAVIDEO

- **Default editing style:** every edit, remake or recut in this repo uses the voidfx style and **starts from the standard edit `edits/new_person` (Dexter × New Person, Same Old Mistakes v6, user-approved; the earlier standard was `edits/hold_on`)** via `edit/tools/new_edit.py`; edits of powered/speedster characters start from `edits/flash` (Flash × Slaughter House v3, `--from flash`), which the user approved as adequate but below the quality they're aiming for, so aim higher. Load the `voidfx-edit` skill (`.claude/skills/voidfx-edit/SKILL.md`) first and follow its rules, workflow and QA checklist. Only deviate when the user asks; if they change a preference, update the skill's rules and its preference log.
- Reverse-engineering a new reference edit: use the rea-video tool (https://github.com/JuneYaooo/rea-video), then record anything new in `docs/style-breakdown.md` or the skill.
- Stored scene packs and reference edits: see `scenepacks/README.md`. Fetch them with `bash scenepacks/fetch.sh` when `media/` is empty, since cloud containers start fresh.
- Never commit footage, music or renders. `media/`, `edit/clips/` and `edit/out/` are gitignored.
