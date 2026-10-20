#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Configuration block
# Edit only this section for common use
# =============================================================================

TASKS=(scf)

# Extra INCAR settings for band/dos/wannier, pay attention to parameters like ISMEAR and NBANDS
BAND_OVERRIDES=("ICHARG=11" "LORBIT=12")
DOS_OVERRIDES=("ICHARG=11")
WANNIER_OVERRIDES=("ICHARG=11" "NCORE=1" "LWANNIER90=.TRUE.")

# DOS k-point multiplier relative to scf.
# Each k-point mesh number is calculated as ceil(scf_value * multiplier).
KPOINT_MULTIPLIER_DOS=2.0

# For 2D materials, always set kz = 1.
KPOINTS_2D_ONLY=1

# Default environment setup for scf/band/dos.
DEFAULT_ENV_SOURCE='source /softwares/bash_scripts/loadmodule.sh'
DEFAULT_ENV_LOAD='module load vasp/5.4.4-intel'

# Environment setup for wannier task.
WANNIER_ENV_LOAD='module load vasp/5.4.4-intel-wannier90'

# Submission command
SUBMIT_CMD='mpirun -np 32 vasp_ncl > log 2>&1'
# =============================================================================


validate_task() {
  case "$1" in
    scf|band|dos|wannier) ;;
    *) echo "Error: unknown task '$1'. Use scf, band, dos, or wannier in TASKS." >&2; exit 1 ;;
  esac
}

run_in_task_env() {
  local task="$1"
  local workdir="$2"

  (
    cd "$workdir"
    case "$task" in
      wannier)
        eval "$WANNIER_ENV_LOAD"
        ;;
      *)
        eval "$DEFAULT_ENV_SOURCE"
        eval "$DEFAULT_ENV_LOAD"
        ;;
    esac
    eval "$SUBMIT_CMD"
    # check_convergence "$PWD"
  )
}

normalize_task_order() {
  local ordered=() target task

  for target in scf band dos wannier; do
    for task in "${TASKS[@]}"; do
      if [ "$task" = "$target" ]; then
        ordered+=("$task")
      fi
    done
  done

  TASKS=("${ordered[@]}")
}

apply_incar_overrides() {
  local incar_file="$1"
  shift
  local pair key value tmp
  tmp="${incar_file}.tmp"
  cp "$incar_file" "$tmp"

  for pair in "$@"; do
    key="${pair%%=*}"
    value="${pair#*=}"
    if [ -z "$key" ] || [ -z "$value" ] || [ "$key" = "$pair" ]; then
      echo "Error: invalid override '$pair'. Use KEY=value format." >&2
      rm -f "$tmp"
      exit 1
    fi
    if grep -Eq "^[[:space:]]*${key}[[:space:]=]" "$tmp"; then
      sed -i -E "s|^[[:space:]]*${key}[[:space:]=].*|${key} = ${value}|" "$tmp"
    else
      printf '\n%s = %s\n' "$key" "$value" >> "$tmp"
    fi
  done

  mv "$tmp" "$incar_file"
}

extract_incar_value() {
  local file="$1"
  local key="$2"
  awk -v k="$key" '
    BEGIN { IGNORECASE=1 }
    $0 ~ "^[[:space:]]*" k "[[:space:]=]" {
      line=$0
      sub(/!.*/, "", line)
      sub(/#.*/, "", line)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)
      split(line, a, /[[:space:]=]+/)
      for (i=1; i<=length(a); i++) if (a[i] == k) { print a[i+1]; exit }
    }
  ' "$file"
}

extract_doscar_fermi() {
  local file="$1"
  awk 'NR==6 { print $4; exit }' "$file"
}

patch_doscar_fermi() {
  local src="$1"
  local dst="$2"
  local fermi

  [ -f "$src" ] || { echo "Error: '$src' not found." >&2; exit 1; }
  [ -f "$dst" ] || { echo "Error: '$dst' not found." >&2; exit 1; }

  fermi="$(extract_doscar_fermi "$src")"
  [ -n "$fermi" ] || { echo "Error: failed to read Fermi level from '$src'." >&2; exit 1; }

  FERMI_VALUE="$fermi" perl -0pi -e '
    my $fermi = $ENV{FERMI_VALUE};
    my @lines = split /\n/, $_, -1;
    die "Error: DOSCAR has fewer than 6 lines\n" if @lines < 6;

    my $line = $lines[5];
    if ($line =~ /^(\s*\S+)(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)(.*)$/) {
      my ($c1, $s1, $c2, $s2, $c3, $s3, $oldfermi, $s4, $c5, $tail) = ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10);
      $lines[5] = $c1 . $s1 . $c2 . $s2 . $c3 . $s3 . $fermi . $s4 . $c5 . $tail . " #" . $oldfermi;
    } else {
      die "Error: unexpected DOSCAR line 6 format\n";
    }

    $_ = join("\n", @lines);
  ' "$dst"
}
check_convergence() {
  local workdir="$1"
  local outcar="${workdir}/OUTCAR"

  if [ ! -f "$outcar" ]; then
    echo "Error: OUTCAR not found in '$workdir'." >&2
    exit 1
  fi

  if grep -qiE 'aborting loop because EDIFF is reached' "$outcar"; then
    return 0
  fi

  echo "Error: VASP in '$workdir' did not reach the requested accuracy." >&2
  exit 1
}

prepare_scf() {
  for f in INCAR POSCAR POTCAR KPOINTS; do
    [ -f "$f" ] || { echo "Error: '$f' not found in current directory." >&2; exit 1; }
  done

  mkdir -p scf
  cp INCAR POSCAR POTCAR KPOINTS scf/
  run_in_task_env scf scf
}

prepare_band() {
  [ -d scf ] || { echo "Error: 'scf' directory not found. Run scf first." >&2; exit 1; }
  [ -f KPATH.in ] || { echo "Error: 'KPATH.in' not found in current directory." >&2; exit 1; }
  [ -f scf/OUTCAR ] || { echo "Error: 'scf/OUTCAR' not found. band requires converged scf results." >&2; exit 1; }

  mkdir -p band
  cp scf/CHG* scf/INCAR scf/POSCAR scf/POTCAR scf/KPOINTS band/ 2>/dev/null || true
  cp KPATH.in band/KPOINTS

  apply_incar_overrides band/INCAR "${BAND_OVERRIDES[@]}"

  run_in_task_env band band
  patch_doscar_fermi scf/DOSCAR band/DOSCAR
  (
    cd band
    printf '211\n' | vaspkit
  )
}

scale_kpoints_file() {
  local src="$1"
  local dst="$2"
  local factor="$3"
  local kz_only="$4"

  awk -v f="$factor" -v kz_only="$kz_only" '
    function ceil(x) { return (x == int(x)) ? int(x) : int(x) + 1 }
    NR <= 3 { print; next }
    NR == 4 {
      if ($1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/) {
        kz = kz_only ? 1 : ceil($3 * f)
        printf "%d %d %d\n", ceil($1 * f), ceil($2 * f), kz
      } else {
        print
      }
      next
    }
    { print }
  ' "$src" > "$dst"
}

prepare_dos() {
  [ -d scf ] || { echo "Error: 'scf' directory not found. Run scf first." >&2; exit 1; }
  [ -f scf/OUTCAR ] || { echo "Error: 'scf/OUTCAR' not found. dos requires converged scf results." >&2; exit 1; }

  mkdir -p dos
  cp scf/CHG* scf/INCAR scf/POSCAR scf/POTCAR scf/KPOINTS dos/ 2>/dev/null || true

  apply_incar_overrides dos/INCAR "${DOS_OVERRIDES[@]}"

  scale_kpoints_file dos/KPOINTS dos/KPOINTS.scaled "$KPOINT_MULTIPLIER_DOS" "$KPOINTS_2D_ONLY"
  mv dos/KPOINTS.scaled dos/KPOINTS

  run_in_task_env dos dos
  patch_doscar_fermi scf/DOSCAR dos/DOSCAR
  (
    cd dos
    printf '111\n' | vaspkit
  )
}

prepare_wannier() {
  [ -d scf ] || { echo "Error: 'scf' directory not found. Run scf first." >&2; exit 1; }
  [ -f scf/OUTCAR ] || { echo "Error: 'scf/OUTCAR' not found. wannier requires converged scf results." >&2; exit 1; }
  [ -f wannier90.win ] || { echo "Error: 'wannier90.win' not found in current directory." >&2; exit 1; }

  mkdir -p wannier
  cp scf/CHG* scf/INCAR scf/POSCAR scf/POTCAR scf/KPOINTS wannier/ 2>/dev/null || true
  cp wannier90.win wannier/

  apply_incar_overrides wannier/INCAR "${WANNIER_OVERRIDES[@]}"

  run_in_task_env wannier wannier
}

for task in "${TASKS[@]}"; do
  validate_task "$task"
done

normalize_task_order

for task in "${TASKS[@]}"; do
  case "$task" in
    scf) prepare_scf ;;
    band) prepare_band ;;
    dos) prepare_dos ;;
    wannier) prepare_wannier ;;
  esac
done

