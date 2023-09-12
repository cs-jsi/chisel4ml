.PHONY = protobuf clean

PROTOC = python -m grpc_tools.protoc --proto_path=chisel4ml/lbir/. \
									 --python_out=chisel4ml/lbir/. \
									 --grpc_python_out=chisel4ml/lbir/. \
									 --mypy_out=chisel4ml/lbir/. \
									 -I=scalapb/scalapb.proto

SRCS := $(wildcard chisel4ml/lbir/*.proto)
BINS := $(SRCS:%.proto=%_pb2.py)
BINS += $(SRCS:%.proto=%_pb2.pyi)
BINS += $(SRCS:%.proto=%_pb2_grpc.py)

protobuf: ${BINS}

%_pb2.py %_pb2.pyi %_pb2_grpc.py: %.proto
	${PROTOC} $<


clean:
	rm ${BINS}
