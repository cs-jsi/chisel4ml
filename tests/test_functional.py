
def test_functional_compatibility(accelerator_qonnx):
    accelerator, qonnx_model = accelerator_qonnx
    if accelerator == "my_accel1":
        assert qonnx_model == 1
    elif accelerator == "my_accel2":
        assert qonnx_model == 2
    else:
        assert False



        