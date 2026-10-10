import json
import logging
import os
import shutil
import joblib
from datetime import datetime
from housing import config

logger = logging.getLogger(__name__)


def save_version(pipeline, metrics: dict) -> str:
    version_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    model_path = config.VERSION_DIR / f"model_{version_id}.joblib"
    metrics_path = config.VERSION_DIR / f"metrics_{version_id}.json"

    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    logger.info("Saved new model version: %s", version_id)
    return version_id


def get_production_metrics() -> dict | None:
    if not config.PRODUCTION_METRICS_PATH.exists():
        return None
    return json.loads(config.PRODUCTION_METRICS_PATH.read_text())


def _copy_atomically(src, dest) -> None:
    # copy to a temp file next to dest, then swap it in, so a failed copy
    # never leaves a half-written production file
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.copy(src, tmp)
    os.replace(tmp, dest)


def promote(version_id: str) -> None:
    model_path = config.VERSION_DIR / f"model_{version_id}.joblib"
    metrics_path = config.VERSION_DIR / f"metrics_{version_id}.json"

    for path in (model_path, metrics_path):
        if not path.exists():
            raise FileNotFoundError(
                f"No such version: {version_id} (looked for {path})")

    _copy_atomically(model_path, config.PRODUCTION_MODEL_PATH)
    _copy_atomically(metrics_path, config.PRODUCTION_METRICS_PATH)

    logger.info("Promoted version %s to production.", version_id)


def promote_if_better(version_id: str, candidate_metrics: dict) -> bool:
    production_metrics = get_production_metrics()

    if production_metrics is None:
        logger.info(
            "No existing production model -- promoting %s as the first one.", version_id)
        promote(version_id)
        return True

    r2_change = candidate_metrics["r2_score"] - production_metrics["r2_score"]

    if r2_change < -config.MAX_ACCEPTABLE_R2_DROP:
        logger.warning(
            "REJECTED version %s: R^2 would drop by %.4f (from %.4f to %.4f), "
            "exceeding the allowed threshold of %.4f. Production model unchanged.",
            version_id, -r2_change, production_metrics["r2_score"],
            candidate_metrics["r2_score"], config.MAX_ACCEPTABLE_R2_DROP,
        )
        return False

    logger.info(
        "Promoting version %s: R^2 %.4f -> %.4f (change: %+.4f)",
        version_id, production_metrics["r2_score"], candidate_metrics["r2_score"], r2_change,
    )
    promote(version_id)
    return True


def list_versions() -> list[dict]:
    versions = []
    for metrics_file in sorted(config.VERSION_DIR.glob("metrics_*.json"), reverse=True):
        version_id = metrics_file.stem.replace("metrics_", "")
        metrics = json.loads(metrics_file.read_text())
        versions.append({"version_id": version_id, **metrics})
    return versions


def rollback_to(version_id: str) -> None:
    logger.info("Rolling back production to version %s.", version_id)
    promote(version_id)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    if len(sys.argv) < 2 or sys.argv[1] not in {"list", "rollback"}:
        print("Usage:")
        print("  python -m housing.ml.model_registry list                 # show all versions")
        print("  python -m housing.ml.model_registry rollback <version_id> # promote an older version")
        sys.exit(1)

    if sys.argv[1] == "list":
        current = get_production_metrics()
        print(f"Current production metrics: {current}\n")

        for v in list_versions():
            print(
                f"  {v['version_id']}  |  R^2={v['r2_score']:.4f}  RMSE={v['rmse']:,.0f}")

    elif sys.argv[1] == "rollback":
        if len(sys.argv) < 3:
            print("Please provide a version_id, e.g.:")
            print("  python -m housing.ml.model_registry rollback 20260921_192455")
            sys.exit(1)
        rollback_to(sys.argv[2])
        print(
            f"Rolled back to version {sys.argv[2]}. This is now the production model.")
