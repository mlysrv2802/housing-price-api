import logging
from datetime import datetime, timedelta

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

from housing import config
from housing.data import fetch_new_data
from housing.ml import model_registry
from housing.ml import train
from housing.data.synthetic import data_generator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def bootstrap_if_needed() -> None:
    if config.PRODUCTION_MODEL_PATH.exists():
        logger.info("Production model already exists -- skipping bootstrap.")
        return

    logger.info("No production model found -- bootstrapping one now (API not required for this step).")
    # load normal data (non messy for the first time)
    if not config.RAW_DATA_PATH.exists():
        logger.info("No training data found either -- generating some first.")
        data_generator.main()

    train.main()
    logger.info("Bootstrap complete -- production model now exists.")


def fetch_and_retrain_job() -> None:
    # Any failure in one cycle (bad data, training error, ...) is logged and
    # swallowed so it can't crash the scheduler -- including the immediate
    # first run, which happens outside APScheduler's own error handling.
    try:
        _run_fetch_and_retrain()
    except Exception:
        logger.exception("Fetch-and-retrain cycle failed -- will retry on next scheduled tick.")


def _run_fetch_and_retrain() -> None:
    try:
        rows_added = fetch_new_data.main()
    except SystemExit:
        # fetch_new_data.main() calls sys.exit(1) on a hard failure (API down,
        # timeout). Catch it here so ONE bad run doesn't kill the whole
        # scheduler process -- the next scheduled tick should still fire.
        logger.error("Fetch step failed this run -- will retry on next scheduled tick.")
        return

    if not rows_added:
        logger.info("No new data this run -- skipping retrain.")
        return

    logger.info("%d new rows added -- triggering retrain.", rows_added)

    previous_metrics = model_registry.get_production_metrics()
    promoted = train.main()
    new_metrics = model_registry.get_production_metrics()

    if not promoted:
        logger.info(
            "Retrain ran, but production model is UNCHANGED (candidate was "
            "rejected by model_registry -- see the WARNING above for why)."
        )

    elif previous_metrics:
        logger.info(
            "Production model UPDATED. R^2: %.4f -> %.4f (%+.4f)",
            previous_metrics["r2_score"], new_metrics["r2_score"],
            new_metrics["r2_score"] - previous_metrics["r2_score"],
        )
    else:
        logger.info("First production model created. R^2: %.4f", new_metrics["r2_score"])


if __name__ == "__main__":
    bootstrap_if_needed()
    scheduler = BlockingScheduler()
    INTERVAL_MINUTES = 60
    scheduler.add_job(
        fetch_and_retrain_job,
        trigger=IntervalTrigger(minutes=INTERVAL_MINUTES),
        id="fetch_and_retrain",
        next_run_time=datetime.now() + timedelta(minutes=INTERVAL_MINUTES)
    )

    logger.info("Scheduler started. Job will run every %d minutes. Press Ctrl+C to stop.", INTERVAL_MINUTES)
    logger.info("Running one job immediately so you can see it work right now...")
    fetch_and_retrain_job()  # run once immediately for demo visibility

    try:
        scheduler.start()
    except(KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
