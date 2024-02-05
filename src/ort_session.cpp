#include "ort_session.hpp"

#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <godot_cpp/templates/vector.hpp>

#include<cassert>

#include <onnxruntime_cxx_api.h>

using namespace godot;

void OnnxSession::_bind_methods()
{
	// ClassDB::bind_method(D_METHOD("hello_singleton"), &OnnxSession::hello_singleton);
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
		ret_dimensions.push_back(x);
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
		ret_dimensions.push_back(x);
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
		for (size_t i=0; i<arr.size();i++)
		{
			Variant this_arr = arr[i];
			// input should be PACKED_FLOAT32_ARRAY
			ERR_FAIL_COND_V_MSG(this_arr.get_type() != Variant::PACKED_FLOAT32_ARRAY,NULL, "OnnxSession input data must be PackedFloat32Array or an Array of them");
			PackedFloat32Array as_arr = static_cast<PackedFloat32Array >(this_arr);
			inputs.push_back(as_arr);
		}
	}
	else if (type == Variant::PACKED_FLOAT32_ARRAY)
	{
		inputs.push_back((PackedFloat32Array)(input));
	}
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
		for(auto i:outputs){
			Variant v=Variant(i);
			a.append(v);
		}
		retval = a;
	}
	return retval;
}

Vector<PackedFloat32Array> OnnxSession::_run_internal(Vector<PackedFloat32Array> &inputs)
{
	Ort::Session &session = *m_session;
	const size_t num_input_nodes = session.GetInputCount();
	ERR_FAIL_COND_V_MSG(inputs.size() != num_input_nodes,Vector<PackedFloat32Array>(), "Wrong number of inputs for OnnxSession run");
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
		ERR_FAIL_COND_V_MSG(input.size() != element_count, Vector<PackedFloat32Array>(),"Input has incorrect size in OnnxSession.run");

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
		auto output_name = session.GetInputNameAllocated(idx, allocator);
		output_node_names.push_back(output_name.get());
		output_names_ptr.push_back(std::move(output_name));
	}
	auto run_options=Ort::RunOptions();
	auto output_tensors =
		session.Run(run_options, input_node_names.data(), in_tensors.data(), in_tensors.size(), output_node_names.data(), output_node_names.size());
	assert(output_tensors.size() == num_output_nodes && output_tensors.front().IsTensor());

	Vector<PackedFloat32Array> out_vals;
	for (auto it=output_tensors.begin(); it!=output_tensors.end(); ++it)
	{
		Ort::Value& out_tensor = *it;
		size_t count = out_tensor.GetCount();
		float *floatarr = out_tensor.GetTensorMutableData<float>();
		PackedFloat32Array new_array;
		new_array.resize(count);
		memcpy(new_array.ptrw(), floatarr, sizeof(float) * count);
		out_vals.push_back(new_array);
	}
	return out_vals;
}
