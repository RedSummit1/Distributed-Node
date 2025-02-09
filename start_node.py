import argparse
import logging
from node import Node
from model import Model
import asyncio
import aiozmq
# import traceback

MODELEX_BANNER = """
░▒▓██████████████▓▒░ ░▒▓██████▓▒░░▒▓███████▓▒░░▒▓████████▓▒░▒▓█▓▒░      ░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓██████▓▒░ ░▒▓█▓▒░      ░▒▓██████▓▒░  ░▒▓██████▓▒░
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░
░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░░▒▓██████▓▒░░▒▓███████▓▒░░▒▓████████▓▒░▒▓████████▓▒░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░

"""


async def main():
    # print(MODELEX_BANNER)
    # print("\nInitializing ModelEx Node...\n")

    # Find and parse the values from system
    parser = argparse.ArgumentParser(
        description="Start a node with specified parameters."
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the node to"
    )
    parser.add_argument(
        "--admin_rest_port", type=int, default=8080, help="HTTP port for the node"
    )
    parser.add_argument(
        "--p2p_port", type=int, default=4443, help="TLS port for the node"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        filename="node.log",
    )

    logger = logging.getLogger(__name__)
    node = None

    try:
        # Create the node with command-line arguments
        node = Node(
            host=args.host, admin_rest_port=args.admin_rest_port, p2p_port=args.p2p_port
        )

        # Start the node
        await node.run()

    # except KeyboardInterrupt:
    #    logger.info("Received keyboard interrupt. Shutting down...")
    # except Exception as e:
    #    logger.error(f"Error in main: {str(e)}")
    #    logger.debug(traceback.format_exc())
    finally:
        logger.info("Command loop has ended. Stopping the node...")


#        if node:
#            try:
#                node.stop()
#                logger.info("Node has been stopped.")
#            except Exception as e:
#                logger.error(f"Error stopping node: {str(e)}")
#                #logger.debug(traceback.format_exc())
#

if __name__ == "__main__":
    asyncio.run(main())
