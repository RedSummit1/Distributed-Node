# P2P Node Project

This project implements a peer-to-peer (P2P) node system with secure communication using SSL/TLS. Each node can connect to other nodes, exchange public keys, and send signed messages.

## Features

- Secure communication using SSL/TLS
- Public key exchange between peers
- Message signing and verification
- HTTP API for node control
- Command-line interface for node interaction

## Prerequisites

- Python 3.7+
- pip (Python package manager)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/RedSummit1/node-jonathan
   cd node-jonathan
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running a Node

1.  Start the node using the `start_node.py` script. You can customize the host and port settings via command-line arguments:
    ```bash
    python start_node.py --host 0.0.0.0 --admin_rest_port 8080 --p2p_port 4443
    ```

2.  **Admin Console:** Once started, the node will output the login screen for the admin console:
    ```
    Admin Console Login
    -----------------
    Username:
    Password:
    ```
    *   Use `admin` as the username and `admin123` as the password to login.

3.  **Using the Admin Console:**

    The admin console provides several commands for node management. Here are some of the essential commands:

    *   `connect <host> <port>`: Connect to another node.
    *   `peers`: List connected peers.
    *   `send <message>`: Send a message to the first connected peer.
    *   `show_model`: Display detailed information about the stored encrypted model.
    *   `tune` or `tune <peer_index> <model_index>`: Initiate a fine-tuning of a local model, or send a model to a remote peer for fine-tuning.
    *   `test_model <text>` or `test_model <model_index> <text>`: Perform sentiment analysis using an available model.
    *   `list_clients`: Show active client connections.
    *   `price <amount>`: Set token price for fine-tuning.
    *   `wallet_balance`: Show node and client token balance.
    *   `init_wallet <amount>`: Add initial tokens to the client wallet.
    *   `escrow_status <escrow_id>`: Check status for specific escrow IDs.
    *   `list_escrows`: List all available escrows.
    *   `list_models`: List all loaded models with their status and SHA256 hash of their weights.
    *   `select_model <index>`: Set a particular model to be the active one.
    *   `model_info`: View details on the current active model.
    *   `exit`: Close the admin console.
    *   `help`: Show available admin commands.

## Security Note

This implementation uses self-signed certificates for SSL/TLS. In a production environment, you should use properly signed certificates from a trusted Certificate Authority.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

[Specify your license here, e.g., MIT, GPL, etc.]
