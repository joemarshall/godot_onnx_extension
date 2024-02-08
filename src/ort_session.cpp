#include "ort_session.hpp"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/templates/vector.hpp>

#include <cassert>
#include <iostream>

#ifdef __MINGW64__
#define _stdcall __stdcall
#endif

#include <onnxruntime_cxx_api.h>

using namespace godot;

void OnnxSession::_bind_methods()
{
	// Variant OnnxSession::run(Variant input);

	// uint32_t num_inputs();
	// PackedInt64Array input_shape(uint32_t idx);
	// String input_name(uint32_t idx);

	// uint32_t num_outputs();
	// PackedInt64Array output_shape(uint32_t idx);
	// String output_name(uint32_t idx);

	ClassDB::bind_method(D_METHOD("run"), &OnnxSession::run);
	ClassDB::bind_method(D_METHOD("num_inputs"), &OnnxSession::num_inputs);
	ClassDB::bind_method(D_METHOD("input_shape"), &OnnxSession::input_shape);
	ClassDB::bind_method(D_METHOD("input_name"), &OnnxSession::input_name);
	ClassDB::bind_method(D_METHOD("num_outputs"), &OnnxSession::num_outputs);
	ClassDB::bind_method(D_METHOD("output_shape"), &OnnxSession::output_shape);
	ClassDB::bind_method(D_METHOD("output_name"), &OnnxSession::output_name);
}

OnnxSession::OnnxSession() : m_session(NULL), RefCounted()
{
}

void OnnxSession::connectOnnxSession(Ort::Session *session)
{
	m_session = session;
}

OnnxSession::~OnnxSession()
{
	if (m_session != NULL)
	{
		// session object should clean itself up on destruction
		delete m_session;
		m_session = NULL;
	}
}

uint32_t OnnxSession::num_inputs()
{
	const Ort::Session &session = *m_session;
	return (uint32_t)session.GetInputCount();
}

PackedInt64Array OnnxSession::input_shape(uint32_t idx)
{
	ERR_FAIL_INDEX_V_MSG(idx, num_inputs(), PackedInt64Array(), vformat("Input index: %d is out of range (0--%d)", idx, num_inputs()));
	const Ort::Session &session = *m_session;
	PackedInt64Array ret_dimensions;

	auto type_info = session.GetInputTypeInfo(idx);
	auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
	auto input_node_dims = tensor_info.GetShape();

	for (auto x : input_node_dims)
	{
		ret_dimensions.push_back(x);
	}
	return ret_dimensions;
}

String OnnxSession::input_name(uint32_t idx)
{
	ERR_FAIL_INDEX_V_MSG(idx, num_inputs(), String(), vformat("Input index: %d is out of range (0--%d)", idx, num_inputs()));
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

PackedInt64Array OnnxSession::output_shape(uint32_t idx)
{
	ERR_FAIL_INDEX_V_MSG(idx, num_outputs(), PackedInt64Array(), vformat("Output index: %d is out of range (0--%d)", idx, num_inputs()));
	const Ort::Session &session = *m_session;
	PackedInt64Array ret_dimensions;

	auto type_info = session.GetOutputTypeInfo(idx);
	auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
	auto output_node_dims = tensor_info.GetShape();

	for (auto x : output_node_dims)
	{
		ret_dimensions.push_back(x);
	}
	return ret_dimensions;
}

String OnnxSession::output_name(uint32_t idx)
{
	ERR_FAIL_INDEX_V_MSG(idx, num_outputs(), String(), vformat("Output index: %d is out of range (0--%d)", idx, num_inputs()));
	const Ort::Session &session = *m_session;

	Ort::AllocatorWithDefaultOptions allocator;
	auto output_name = session.GetOutputNameAllocated(idx, allocator);
	return String(output_name.get());
}

Variant OnnxSession::run(Variant input)
{
	// input is either: PackedFloatArray (for single input), or Array<PackedFloatArray> for multi input
	Vector<PackedFloat32Array> inputs;
	auto type = input.get_type();
	if (type == Variant::ARRAY)
	{
		// multi-input
		// n.b. don't support plain arrays for single input
		Array arr = input;
		for (size_t i = 0; i < arr.size(); i++)
		{
			Variant this_arr = arr[i];
			// input should be PACKED_FLOAT32_ARRAY
			ERR_FAIL_COND_V_MSG(this_arr.get_type() != Variant::PACKED_FLOAT32_ARRAY, NULL, "OnnxSession input data must be PackedFloat32Array or an Array of them");
			PackedFloat32Array as_arr = static_cast<PackedFloat32Array>(this_arr);
			inputs.push_back(as_arr);
		}
	}
	else if (type == Variant::PACKED_FLOAT32_ARRAY)
	{
		inputs.push_back((PackedFloat32Array)(input));
	}
	else
	{
		ERR_FAIL_V_MSG(Variant(), "Input needs to be either a single PackedFloat32Array, or an Array of them");
	}
	try
	{
		auto outputs = _run_internal(inputs);
		Variant retval;
		if (outputs.size() == 1)
		{
			// if model has 1 output, return a packedfloat32array
			retval = Variant(outputs[0]);
		}
		else
		{
			// else return array of packedfloat32array
			// TODO: will this cast even work?
			auto a = Array();
			for (auto i : outputs)
			{
				Variant v = Variant(i);
				a.append(v);
			}
			retval = a;
		}
		return retval;
	}
	catch (const std::exception &e)
	{
		std::cerr << "Exc" <<e.what() << '\n';
	}
	return Variant();
}

Vector<PackedFloat32Array> OnnxSession::_run_internal(Vector<PackedFloat32Array> &inputs)
{
	Ort::Session &session = *m_session;
	const size_t num_input_nodes = session.GetInputCount();
	ERR_FAIL_COND_V_MSG(inputs.size() != num_input_nodes, Vector<PackedFloat32Array>(), "Wrong number of inputs for OnnxSession run");
	std::vector<Ort::Value> in_tensors;

	std::vector<const char *> input_node_names;
	// this second vector is used because otherwise output_name will be deallocated
	// at end of loop iteration below, whereas we want it deallocated after calling session.Run
	std::vector<Ort::AllocatedStringPtr> input_node_names_ptr;
	Ort::AllocatorWithDefaultOptions allocator;
	for (size_t idx = 0; idx < num_input_nodes; idx++)
	{
		auto input_name = session.GetInputNameAllocated(idx, allocator);
		input_node_names.push_back(input_name.get());
		input_node_names_ptr.push_back(std::move(input_name));
	}

	for (size_t idx = 0; idx < num_input_nodes; idx++)
	{
		auto type_info = session.GetInputTypeInfo(idx);
		auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
		auto input_node_dims = tensor_info.GetShape();
		auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		auto element_count = tensor_info.GetElementCount();

		auto input = inputs[idx];
		ERR_FAIL_COND_V_MSG(input.size() != element_count, Vector<PackedFloat32Array>(), vformat("Input has incorrect size in OnnxSession.run %d, expected %d", input.size(), element_count));

		Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input.ptrw(), element_count,
																  input_node_dims.data(), input_node_dims.size());
		assert(input_tensor.IsTensor());
		in_tensors.push_back(std::move(input_tensor));
	}

	const size_t num_output_nodes = session.GetOutputCount();
	std::vector<const char *> output_node_names;
	// this second vector is used because otherwise output_name will be deallocated
	// at end of loop iteration below, whereas we want it deallocated after calling session.Run
	std::vector<Ort::AllocatedStringPtr> output_names_ptr;
	for (size_t idx = 0; idx < num_output_nodes; idx++)
	{
		auto output_name = session.GetOutputNameAllocated(idx, allocator);
		output_node_names.push_back(output_name.get());
		output_names_ptr.push_back(std::move(output_name));
	}
	auto run_options = Ort::RunOptions();
	auto output_tensors =
		session.Run(run_options, input_node_names.data(), in_tensors.data(), in_tensors.size(), output_node_names.data(), output_node_names.size());
	assert(output_tensors.size() == num_output_nodes && output_tensors.front().IsTensor());

	Vector<PackedFloat32Array> out_vals;
	for (auto it = output_tensors.begin(); it != output_tensors.end(); ++it)
	{
		Ort::Value &out_tensor = *it;
		auto tensorShape = out_tensor.GetTensorTypeAndShapeInfo();
		size_t count = tensorShape.GetElementCount();
		float *floatarr = out_tensor.GetTensorMutableData<float>();
		PackedFloat32Array new_array;
		new_array.resize(count);
		memcpy(new_array.ptrw(), floatarr, sizeof(float) * count);
		out_vals.push_back(new_array);
	}
	return out_vals;
}
