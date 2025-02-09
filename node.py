import pickle
import torch
from web3 import Web3
from datasets import Dataset
from eth_account import Account
from web3.gas_strategies.time_based import medium_gas_price_strategy
import os
from dotenv import load_dotenv
import sys
import io
import hashlib
import uuid
import uvicorn
import json
import logging
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
import datetime
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from generate_certificate import generate_self_signed_cert
from server import create_server
from communication_protocal import P2PNetwork
from model import Model


# TODO Understand why we need this code for the errors to show
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# TODO Understand why this logger file is needed in order to show the terminal
logger = logging.getLogger(__name__)
CONTRACT_ADDRESS = "0x242c0EbD4F2219ceaB39B52374DdFd4b0ADe4af1"


class Node:
    def __init__(
        self,
        host="0.0.0.0",
        admin_rest_port=8080,
        p2p_port=4443,
        client_base_port=5000,
        max_clients=10,
        thread_pool_size=30,
    ):
        # Create logger that is scoped in node
        self.logger = logging.getLogger(f"{__name__}.Node")

        load_dotenv()

        # Basic configuration
        self.host = host
        self.admin_rest_port = admin_rest_port
        self.p2p_port = p2p_port
        self.node_address = f"node_{self.p2p_port}"

        # Get the name of the cert
        self.cert_path = "cert.pem"

        # Initialize thread pool (using standard library ThreadPoolExecutor)
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)

        # Receive uvicorn and P2P server
        self.server = create_server(self, self.host, self.admin_rest_port)
        self.comm_port = P2PNetwork(self, self.p2p_port)

        # Get the certificate
        cert = generate_self_signed_cert("localhost", self, [self.host])
        self.public_key = self.read_private_key().public_key()

        # Read models from working directory
        self.read_models()

        # Create Job Schedule
        self.job_schedule = {}

    def read_private_key(self):
        try:
            self.logger.info("Read private key from .env file")
            private_key = os.getenv("SSH_PRIVATE_KEY")

            if not private_key:
                raise ValueError("SSH_PRIVATE_KEY not found in environment variables")

            private_key = private_key.encode("utf-8")

            private_key = serialization.load_ssh_private_key(
                private_key,
                password=None,
            )

            self.logger.info(
                f"Private key {'is instance' if isinstance(private_key, rsa.RSAPrivateKey) else 'is not instance'} of rsa.RSAPrivateKey"
            )
            return private_key

        except Exception as e:
            self.logger.error(f"Failed to read SSH key: {str(e)}")

    def read_models(self):
        self.models = []

        pkl_files = [f for f in os.listdir(".") if f.endswith(".pkl")]

        for pkl_file in pkl_files:
            try:
                with open(pkl_file, "rb") as f:
                    model = pickle.load(f)
                    self.models.append(model)
            except Exception as e:
                print(f"Error loading {pkl_file}:{str(e)}")

    def check_auth(auth_code):
        pass

    async def run(self):
        self.stop_event = asyncio.Event()
        asyncio.create_task(self.server.serve())
        asyncio.create_task(self.comm_port.run())
        # asyncio.create_task(self.stop())
        await self.stop_event.wait()

    async def stop(self):
        await asyncio.sleep(2)
        self.stop_event.set()

        # Initialize sentiment models


#
#        # Create ethereum public address for Node
#        #private_key = os.urandom(32)
#        #account = Account.from_key(private_key)
#        #self.address = account.address
#        #print(f"Assigned public address of node:{self.address}")
#        #self.private_key = private_key.hex()
#        #self.load_all_models()
#
#
# Initialize collections and state tracking
# TODO Remove instances of node within the peer
#        self.active_tasks = {}
#        self.task_futures = {}
#        self.jobScheduler = {}
#        # self.current_jobs = {}
#        self.current_jobID = None
#

# Initialize FastAPI

#    def setup_routes(self):
#        @self.app.get("/kill")
#        async def kill_node():
#            return self.stop()
#
#        @self.app.get("/ping")
#        async def ping_node():
#            return {"status": "ok"}
#
#        @self.app.get("/jobScheduler")
#        async def get_jobScheduler():
#            return {"jobSchedule": self.jobScheduler}
#
#        @self.app.get("/currentJobs")
#        async def get_current_jobs():
#            return {"currentJobs": self.current_jobs}
#
#        @self.app.post("/startTraining")
#        async def start_training_endpoint(
#            auth_code: str = Body(), job_id: str = Body()
#        ):
#            """Test endpoint to verify server is running"""
#            self.current_jobID = job_id
#            job = {job_id: {"auth_code": auth_code, "status": "CREATED"}}
#            self.jobScheduler.update(**job)
#
#            def execute_do_tune():
#                try:
#                    # Capture the standard output and standard error to check if an issue is occurring.
#                    # The do_tune is not designed for returning values from an API.
#
#                    old_stdout = sys.stdout
#                    old_stderr = sys.stderr
#
#                    redirected_output = sys.stdout = io.StringIO()
#                    redirected_error = sys.stderr = io.StringIO()
#
#                    self.console.onecmd("tune 0 0")
#
#                    sys.stdout = old_stdout
#                    sys.stderr = old_stderr
#
#                    stdout_contents = redirected_output.getvalue()
#                    stderr_contents = redirected_error.getvalue()
#
#                    if (
#                        "Error" in stdout_contents
#                        or "failed" in stdout_contents.lower()
#                        or stderr_contents != ""
#                    ):
#                        return {
#                            "status": "error",
#                            "message": f"do_tune failed. Standard Output:{stdout_contents} Standard Error:{stderr_contents}",
#                        }
#
#                    return {
#                        "status": "sent",
#                        "message": f"do_tune finished. Standard Output:{stdout_contents}",
#                    }
#                except Exception as e:
#                    return {"status": "error", "message": str(e)}
#
#            future = self.thread_pool.submit(execute_do_tune)
#            return future.result()
#
#        @self.app.get("/info")
#        async def info():
#            model_hashs = []
#            for i, model_data in enumerate(self.models):
#                model = model_data["model"]
#                param_hash = None
#                for name, param in model.state_dict().items():
#                    param_bytes = param.cpu().numpy().tobytes()
#                    param_hash = hashlib.sha256(param_bytes).hexdigest()
#                model_hashs.append({"name": f"Model_{i + 1}", "hashWB": param_hash})
#
#            public_key_pem = self.public_key.public_bytes(
#                encoding=serialization.Encoding.PEM,
#                format=serialization.PublicFormat.SubjectPublicKeyInfo,
#            ).decode("utf-8")
#
#            # List to store dictionaries of file info
#            file_info = []
#
#            # Loop through all files in current directory
#            for filename in os.listdir("."):
#                # Check if file is a CSV
#                if filename.endswith(".csv"):
#                    with open(filename, "rb") as f:
#                        sha256 = hashlib.sha256()
#
#                        # Read in chunks for large files
#                        for chunk in iter(lambda: f.read(4096), b""):
#                            sha256.update(chunk)
#
#                        # Create dictionary and add to list
#                        file_info.append(
#                            {"name": filename, "hashDB": sha256.hexdigest()}
#                        )
#
#            return json.loads(
#                json.dumps(
#                    {
#                        "name": "John's node",
#                        "public_key": public_key_pem,
#                        "ethereum_address": self.address,
#                        "port": self.p2p_port,
#                        "rpcPort": self.admin_rest_port,
#                        "registered": True,
#                        "models": model_hashs,
#                        "data_blocks": file_info,
#                    }
#                )
#            )
#
#        @self.app.get("/getJobStatus/{job_id}")
#        async def get_jobstatus(job_id):
#            try:
#                job = self.jobScheduler[job_id]
#            except KeyError:
#                return {"error": "Job not in scheduler"}
#            return job.get("status")
#
#        @self.app.post("/submitJob")
#        async def update_register(auth_code: str = Body(...)):
#            """
#            Submit a new job to the blockchain and local job scheduler.
#
#            :param auth_code: Authentication code for the job
#            :return: Job submission result or error response
#            """
#            load_dotenv()
#            try:
#                # Generate unique job ID
#                job_id = str(uuid.uuid4())
#                job_id_int = uuid.uuid4().int & (
#                    (1 << 128) - 1
#                )  # Ensure 128-bit integer
#
#                # Get info from /info endpoint
#                async with aiohttp.ClientSession() as session:
#                    async with session.get("http://localhost:8080/info") as response:
#                        info = await response.json()
#
#                # Get the model_hash from the first model
#                model_hash = info["models"][0]["hashWB"]
#
#                # Get data block hash from first data block
#                data_block_hash = info["data_blocks"][0]["hashDB"]
#
#                # Hash the auth code using keccak
#                auth_hash = Web3.keccak(text=auth_code).hex()
#
#                # Convert hashes to bytes32
#                auth_hash_bytes = Web3.to_bytes(hexstr=auth_hash)
#                model_hash_bytes = Web3.to_bytes(hexstr=model_hash)
#                data_block_hash_bytes = Web3.to_bytes(hexstr=data_block_hash)
#
#                # Prepare job metadata
#                now = datetime.datetime.now()
#                TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
#                job = {
#                    job_id: {
#                        "auth_code": auth_code,
#                        "created": now.strftime(TIMESTAMP_FORMAT),
#                        "updated": now.strftime(TIMESTAMP_FORMAT),
#                        "status": "NOT_STARTED",
#                        "modelHash": model_hash,
#                        "dataBlockHash": data_block_hash,
#                    }
#                }
#
#                # Update local job scheduler
#                self.current_jobID = job_id
#                self.jobScheduler.update(**job)
#
#                # Interact with the smart contract
#                # Load private key
#
#                # Initialize web3 client
#                w3 = Web3(
#                    Web3.HTTPProvider("https://eth-holesky.g.alchemy.com/v2/<api-key>")
#                )
#
#                # Get contract object
#                with open("JobScheduler.json", "r") as f:
#                    abi = json.load(f)["abi"]
#                contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=abi)
#
#                # Build transaction
#                tx = contract.functions.createJob(
#                    job_id_int,  # uint128
#                    auth_hash_bytes,  # bytes32
#                    "NOT_STARTED",  # string
#                    model_hash_bytes,  # bytes32
#                    data_block_hash_bytes,  # bytes32
#                ).build_transaction(
#                    {
#                        "from": account.address,
#                        "nonce": w3.eth.get_transaction_count(account.address),
#                        "gas": 200000,  # Adjust based on estimated gas
#                        "gasPrice": w3.eth.gas_price,
#                    }
#                )
#
#                # Sign and send transaction
#                signed_tx = account.sign_transaction(tx)
#                tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
#
#                # Log transaction details
#                self.logger.info(
#                    f"Job submitted successfully. Job ID: {job_id}, Tx Hash: {tx_hash.hex()}"
#                )
#
#                return {"job_id": job_id, "tx_hash": tx_hash.hex()}
#
#            except Exception as e:
#                # Log detailed error
#                self.logger.error(f"Job submission failed: {str(e)}")
#                self.logger.error(traceback.format_exc())
#
#                # Return a structured error response
#                return {"error": "Job submission failed", "details": str(e)}
