.PHONY = protobuf clean

PROTOC = python3 -m grpc_tools.protoc --proto_path=. --python_out=. --grpc_python_out=. --mypy_out=.

SRCS := $(wildcard chisel4ml/lbir/*.proto)
BINS := $(SRCS:%.proto=%_pb2.py) 
BINS += $(SRCS:%.proto=%_pb2.pyi) 
BINS += $(SRCS:%.proto=%_pb2_grpc.py)

all: ${BINS}

%_pb2.py %_pb2.pyi %_pb2_grpc.py: %.proto
	${PROTOC} $<


clean:
	rm ${BINS}
