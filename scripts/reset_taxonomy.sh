#!/usr/bin/env bash
set -euo pipefail

source ~/devel/vibe-semdt/semdt-from-2d-video/cram-env/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CRAM_DIR="$(cd "$SCRIPT_DIR/../../cognitive_robot_abstract_machine" && pwd)"

echo "=== Resetting class taxonomy (simple) ==="
python "$SCRIPT_DIR/reset_class_taxonomy.py" --simple

echo "=== Regenerating ORM ==="
python "$CRAM_DIR/semantic_digital_twin/scripts/generate_orm.py"

echo "=== Done ==="
