import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
import asyncio
import pickle
import json
import argparse
import aiozmq
import zmq
import logging
from zmq.utils.z85 import encode, decode
from model import Model


class P2PNetwork:
    def __init__(self, node, port, host="localhost"):
        self.host = host
        self.port = port
        self.address = f"{host}:{port}"
        self.peers = {}  # {port: socket}
        self.router = None
        self.public_key, self.secret_key = zmq.curve_keypair()
        self.received_public_key = None
        self.received_address = None
        self.node = node
        self.running = True
        self.logger = logging.getLogger(f"{__name__}.P2PNetwork")

    async def setup(self):
        self.router = await aiozmq.create_zmq_stream(
            zmq.ROUTER, bind=f"tcp://*:{self.port}"
        )
        self.router.transport.setsockopt(zmq.CURVE_SERVER, 1)
        self.router.transport.setsockopt(zmq.CURVE_SECRETKEY, self.secret_key)
        self.router.transport.setsockopt(zmq.CURVE_PUBLICKEY, self.public_key)
        print(f"Started on port {self.port}")
        print(f"Public key: {encode(self.public_key)}")

    async def connect_peer(self, port, peer_key, host="localhost"):
        dealer = await aiozmq.create_zmq_stream(
            zmq.DEALER, connect=f"tcp://{host}:{port}"
        )
        dealer.transport.setsockopt(zmq.CURVE_SERVERKEY, peer_key)
        dealer.transport.setsockopt(zmq.CURVE_PUBLICKEY, self.public_key)
        dealer.transport.setsockopt(zmq.CURVE_SECRETKEY, self.secret_key)
        self.peers[port] = dealer
        print(f"Connected to peer {port}")

    async def send(self, message, target_port=None):
        msg = message.encode()
        if target_port:
            if target_port in self.peers:
                self.peers[target_port].write([msg])
                print(f"Sent to peer {target_port}: {message}")
            else:
                print(f"Not connected to peer {target_port}")
        else:
            for port, socket in self.peers.items():
                socket.write([msg])
            print(f"Broadcast: {message}")

    async def send_model(self, model=None, target_port=None, fine_tuned=False):
        if not model:
            model = self.node.models[0]

        if fine_tuned:
            action = b"TRAINED".ljust(8)
        else:
            action = b"TRAIN".ljust(8)

        serialized_model = pickle.dumps(model)
        header = action + self.public_key.ljust(32) + self.address.encode().ljust(128)
        msg = header + serialized_model
        if target_port:
            if target_port in self.peers:
                self.peers[target_port].write([msg])
                print(f"Sent to peer {target_port}: model")
            else:
                print(f"Not connected to peer {target_port}")

    async def receive_messages(self):
        while self.running:
            try:
                msg = await self.router.read()
                self.logger.info("Got the payload")
                if len(msg) >= 1:
                    content = msg[-1]
                    (
                        command,
                        self.received_public_key,
                        self.received_address,
                        serialized_model,
                    ) = (
                        content[:8].strip().decode(),
                        content[8:40],
                        content[40:168].strip().decode(),
                        content[168:].strip(),
                    )
                    self.logger.info("Assigned the variables from the payload")
                    model = pickle.loads(serialized_model)
                    self.logger.info("Pickled model")
                    if command.upper() == "TRAINED":
                        self.logger.info("Final points in if statement")
                        self.node.models.append(model)
                        self.logger.info("Got it!!")
                        break

                    self.logger.info(f"{self.received_address.split(':')}")
                    host_name, port_number = self.received_address.split(":")
                    port_number = int(port_number)
                    await self.connect_peer(
                        int(port_number),
                        self.received_public_key,
                    )
                    self.logger.info(f"Peer in node?:{port_number in self.peers}")
                    await self.fine_tune_model(model)
                    await self.send_model(
                        model=model, target_port=port_number, fine_tuned=True
                    )
                else:
                    print(f"\nReceived: {content.get('model')}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Receive error: {e}")
                await asyncio.sleep(0.1)

    async def fine_tune_model(self, model):
        data = []
        # Get all files in current directory
        for file in os.listdir("."):
            # Check if file has .csv extension
            if file.endswith(".csv"):
                data.append(file)

        model.fine_tune("your_data.csv", output_dir="test_fine_tune", num_epochs=3)

    async def handle_commands(self):
        while self.running:
            try:
                command = await asyncio.get_event_loop().run_in_executor(
                    None,
                    input,
                    "\nCommands: connect <port> <key> | send <port> <msg> | model <port> | broadcast <msg> | quit\n> ",
                )
                parts = command.strip().split()

                if not parts:
                    continue

                if parts[0] == "connect" and len(parts) == 3:
                    await self.connect_peer(int(parts[1]), decode(parts[2]))
                elif parts[0] == "send" and len(parts) >= 3:
                    await self.send(" ".join(parts[2:]), int(parts[1]))
                elif parts[0] == "model":
                    await self.send_model(None, int(parts[1]))
                elif parts[0] == "broadcast" and len(parts) >= 2:
                    await self.send(" ".join(parts[1:]))
                elif parts[0] == "quit":
                    self.running = False
                    break

            except Exception as e:
                print(f"Error: {e}")

    async def run(self):
        await self.setup()
        await asyncio.gather(self.receive_messages(), self.handle_commands())


# if __name__ == "__main__":
#    parser = argparse.ArgumentParser()
#    parser.add_argument("--port", type=int, required=True)
#    args = parser.parse_args()
#
#    logging.basicConfig(level=logging.ERROR)
#    asyncio.run(P2PNetwork(args.port).run())
