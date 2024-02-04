#include "ort_singleton.hpp"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>

#include <onnxruntime_cxx_api.h>

using namespace godot;

void OnnxSession::_bind_methods()
{
	ClassDB::bind_method(D_METHOD("hello_singleton"), &OnnxSession::hello_singleton);
}

OnnxSession::OnnxSession(Ort::Session *in_session) : m_session(in_session)
{
	ERR_FAIL_COND(in_session == NULL);
}

OnnxSession::~OnnxSession()
{
	// session object should clean itself up on destruction
	delete m_session;
	m_session = NULL;
}

uint32_t OnnxSession::num_inputs()
{
	const Ort::Session &session = *m_session;
	return (uint32_t)session.GetInputCount();
}

Vector<int64_t> OnnxSession::input_shape(uint32_t idx)
{
	const Ort::Session &session = *m_session;
	Vector<int64_t> ret_dimensions;

	auto type_info = session.GetInputTypeInfo(idx);
	auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
	auto input_node_dims = tensor_info.GetShape();

	for (auto x : input_node_dims)
	{
		ret_dimensions.push_back(x)
	}
	return ret_dimensions;
}

String OnnxSession::input_name(uint32_t idx)
{
	const Ort::Session &session = *m_session;

	Ort::AllocatorWithDefaultOptions allocator;
	auto input_name = session.GetInputNameAllocated(idx, allocator);
	String input_copy = String(input_name.get());
	return input_copy;
}

uint32_t OnnxSession::num_outputs()
{
	const Ort::Session &session = *m_session;
	return (uint32_t)session.GetOutputCount();
}

Vector<int64_t> OnnxSession::output_shape(uint32_t idx)
{
	const Ort::Session &session = *m_session;
	Vector<int64_t> ret_dimensions;

	auto type_info = session.GetOutputTypeInfo(idx);
	auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
	auto output_node_dims = tensor_info.GetShape();

	for (auto x : output_node_dims)
	{
		ret_dimensions.push_back(x)
	}
	return ret_dimensions;
}

String OnnxSession::output_name(uint32_t idx)
{
	const Ort::Session &session = *m_session;

	Ort::AllocatorWithDefaultOptions allocator;
	auto output_name = session.GetOutputNameAllocated(idx, allocator);
	return String(output_name.get());
}

OnnxSession::set_input_data(Vector<float_t> input, int idx)
{
	// todo: cache this stuff if needed
	auto type_info = session.GetInputTypeInfo(idx);
	auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
	auto input_node_dims = tensor_info.GetShape();

	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	auto element_count = tensor_info.GetElementCount();

	ERR_FAIL_COND_EDMSG(input.size() != element_count, "Wrong number of values to OnnxSession set_input_data");

	auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), element_count,
														input_node_dims.data(), 4);
	assert(input_tensor.IsTensor());
}

OnnxSession::run(Variant input)
{
	// input is either: PackedFloatArray (for single input), or Array<PackedFloatArray> for multi input
	Vector<Vector<float_t>> inputs;
	auto type = input.get_type();
	if (type == Variant::ARRAY)
	{
		// multi-input
		// n.b. don't support plain arrays for single input
		Vector<Variant> arr = input;
		for (auto input : arr)
		{
			// input should be PACKED_FLOAT32_ARRAY
			ERR_FAIL_COND_EDMSG(input.get_type() != Variant::PACKED_FLOAT32_ARRAY, "OnnxSession input data must be PackedFloat32Array or an Array of them");
			inputs.push_back(Vector<float_t>(input));
		}
	}
	else if (type == PACKED_FLOAT32_ARRAY)
	{
		inputs.push_back(Vector<float_t>(input))
	}
	auto outputs = _run_internal(inputs);
	Variant retval;
	if (size(outputs) == 1)
	{
		// if model has 1 output, return a packedfloat32array
		retval = outputs[0];
	}
	else
	{
		// else return array of packedfloat32array
		// TODO: will this cast even work?
		retval = outputs;
	}
}

Vector<Vector<float_t>> OnnxSession::_run_internal(Vector<Vector<float_t>> inputs)
{
	const Ort::Session &session = *m_session;
	const size_t num_input_nodes = session.GetInputCount();
	ERR_FAIL_COND_EDMSG(inputs.size() != num_input_nodes, "Wrong number of inputs for OnnxSession run");
	Vector<Ort::Value> in_tensors;

	for (size_t idx = 0; idx < num_input_nodes; idx++)
	{
		auto type_info = session.GetInputTypeInfo(idx);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
		auto input_node_dims = tensor_info.GetShape();
		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		auto element_count = tensor_info.GetElementCount();

		ERR_FAIL_COND_EDMSG(input.size() != element_count, "Wrong number of values to OnnxSession set_input_data");

		Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), element_count,
																  input_node_dims.data(), 4);
		assert(input_tensor.IsTensor());
		in_tensors.push_back(input_tensor);
	}

	const size_t num_output_nodes = session.GetOutputCount();
	Vector<const char *> output_node_names;
	// this second vector is used because otherwise output_name will be deallocated
	// at end of loop iteration below, whereas we want it deallocated after calling session.Run
	std::vector<Ort::AllocatedStringPtr> output_names_ptr;
	for (size_t idx = 0; idx < num_output_nodes; idx++)
	{
		auto output_name = session.GetInputNameAllocated(i, allocator);
		output_node_names.push_back(output_name.get());
		output_names_ptr.push_back(std::move(output_name));
	}

	auto output_tensors =
		session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), in_tensors.data(), in_tensors.size(), output_node_names.data(), output_node_names.size());
	assert(output_tensors.size() == num_output_nodes && output_tensors.front().IsTensor());

	Vector<Vector<float_t>> out_vals;
	for (auto out_tensor : output_tensors)
	{
		size_t count = out_tensor.count() float *floatarr = out_tensor.GetTensorMutableData<float>();
		Vector<float_t> new_array;
		new_array.resize(count)
			memcpy(new_array.ptrw(), floatarr, sizeof(float_t) * count);
		out_vals.push_back(new_array)
	}
	return out_vals;
}
