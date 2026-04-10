"""Django management command: merge_ghost_town_replay.

Merges master_movement.json + affect_timeline.json + events.json +
social_graph_timeline.json from a ghost town run directory into a single
full_replay.json file that the frontend can load once at startup.

Usage:
    python manage.py merge_ghost_town_replay <sim_code>
    python manage.py merge_ghost_town_replay condition_d_standard_night_seed0_12-agent
"""
from __future__ import annotations

import json
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError


PROJECT_ROOT = Path(__file__).resolve().parents[5]
GHOST_TOWN_OUTPUTS = PROJECT_ROOT / "outputs"

OPTIONAL_FILES = {
    "affect_timeline": "affect_timeline.json",
    "events": "events.json",
    "social_graph_timeline": "social_graph_timeline.json",
    "world_metadata": "world_metadata.json",
    "meta": "meta.json",
}


class Command(BaseCommand):
    help = "Merge ghost town replay artifacts into a single full_replay.json."

    def add_arguments(self, parser):
        parser.add_argument(
            "sim_code",
            type=str,
            help="Run ID / sim code (subdirectory name under outputs/).",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default=None,
            help="Override output directory (defaults to the same run directory).",
        )

    def handle(self, *args, **options):
        sim_code = options["sim_code"]
        run_dir = GHOST_TOWN_OUTPUTS / sim_code
        if not run_dir.exists():
            raise CommandError(f"Run directory not found: {run_dir}")

        move_file = run_dir / "master_movement.json"
        if not move_file.exists():
            raise CommandError(f"master_movement.json not found in {run_dir}")

        self.stdout.write(f"Loading master_movement.json... ", ending="")
        with open(move_file, encoding="utf-8") as fh:
            master_movement = json.load(fh)
        self.stdout.write(f"{len(master_movement)} steps.")

        full_replay: dict = {
            "sim_code": sim_code,
            "master_movement": master_movement,
        }

        for key, filename in OPTIONAL_FILES.items():
            file_path = run_dir / filename
            if file_path.exists():
                self.stdout.write(f"Loading {filename}... ", ending="")
                with open(file_path, encoding="utf-8") as fh:
                    data = json.load(fh)
                full_replay[key] = data
                self.stdout.write("ok.")
            else:
                self.stdout.write(self.style.WARNING(f"  {filename} not found — skipping."))
                full_replay[key] = None

        # Compute step count from master_movement
        full_replay["total_steps"] = len(master_movement)

        # If meta exists, extract condition for frontend label switching
        meta = full_replay.get("meta") or {}
        full_replay["condition"] = meta.get("condition", "")
        full_replay["scenario"] = meta.get("scenario", "")

        output_dir = Path(options["output_dir"]) if options["output_dir"] else run_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "full_replay.json"

        self.stdout.write(f"Writing full_replay.json... ", ending="")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(full_replay, fh, ensure_ascii=False)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        self.stdout.write(self.style.SUCCESS(f"Done. ({size_mb:.1f} MB) → {out_path}"))
