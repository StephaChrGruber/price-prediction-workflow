import logging
import sys
from argparse import Namespace

import data_collection_step
import prep_data_step
import training_step
from data_collection_step import pull_data_from_mongo
from prep_data_step import prep_data
from training_step import train

from util.diagnostics import set_seed, setup_diagnostics, disable_diagnostics

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)-25s] %(asctime)s [%(levelname)-8s]: %(message)s",
    stream=sys.stdout,
    force=True
)

__log = logging.getLogger(__name__)

args: Namespace

def main():
    __log.info("Starting up")
    set_seed(args.seed)
    data_collection_step.args = args
    pull_data_from_mongo()
    prep_data_step.args = args
    prep_data()
    training_step.args = args
    train()

if __name__ == "__main__":
    from util.argument_parser import parse_args
    args = parse_args()
    setup_diagnostics()
    main()
    disable_diagnostics()

