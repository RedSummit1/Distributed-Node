DEFAULT_GLOBAL:= format
.PHONY: customer, provider
format:
	ruff format *.py
customer:format
	python start_node.py
provider:format
	python start_node.py --admin_rest_port 8081 --p2p_port 4444
test:format
	python start_node.py --host 127.0.0.2
test2:format
	python start_node.py --host 127.0.0.2 --admin_rest_port 8081 --p2p_port 4444
