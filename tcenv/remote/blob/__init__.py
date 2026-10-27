"""Independent content-addressed blob channel (P0b).

``tcenv.remote.blob_channel`` is the store, ``tcenv.remote.blob_service`` is the
gRPC adapter and ``tcenv.remote.blob.v1`` holds the generated Protobuf artifacts.
This package is deliberately separate from ``tcenv.remote.v1`` so the six-RPC
Evaluator control plane cannot grow a large-artifact message.
"""
