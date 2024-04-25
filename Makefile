.PHONY = protobuf clean

PROTOC = python -m grpc_tools.protoc --proto_path=chisel4ml/lbir/ \
									 --proto_path=chisel4ml/protoscala/ \
									 --python_out=chisel4ml/lbir/. \
									 --grpc_python_out=chisel4ml/lbir/. \

SRCS := $(wildcard chisel4ml/lbir/*.proto)
SRCS += chisel4ml/protoscala/scalapb/scalapb.proto
BINS := $(SRCS:%.proto=%_pb2.py)
BINS += $(SRCS:%.proto=%_pb2.pyi)
BINS += $(SRCS:%.proto=%_pb2_grpc.py)

protobuf: ${BINS}

%_pb2.py %_pb2.pyi %_pb2_grpc.py: %.proto
	${PROTOC} $<


clean:
	rm -f ${BINS}
